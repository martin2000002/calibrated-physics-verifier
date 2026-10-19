"""Acceptance audit of a migration against a copy taken before it: the velocity convention (WP-G
1.C, A3-A10), the floor crossing of a penetrated bounce row (WP-H, H3 and H8-H13) and the floor
rule of a ball-ball collision (WP-I, I2-I6 and I11).

Every finding is a list of clip ids (or cells) with the numbers that put it there, so the report
says what moved and why rather than only whether a threshold held."""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from physverify.extraction.cache import StageCache
from physverify.physics.measures.r3_collision import kinetic_energy_gain, restitution
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactTable
from physverify.schema.scene import PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.gt_export import DATASET_NAME, final_record
from physverify.synth.kinematics_gt import corrected_cells
from physverify.synth.migrate import (
    COVERS_KEY,
    FLOOR_CROSSING,
    PAIR_COLLISION,
    rerender_clip_ids,
    velocity_migration,
)
from physverify.synth.perturb import load_simulation
from physverify.synth.perturbations._ballistics import impact_velocity
from physverify.synth.perturbations._edit import (
    first_floor_impact,
    floor_impacts,
    gravity_of,
    radius_of,
)
from physverify.synth.perturbations._measure import (
    bounce_crossing,
    bounce_rows,
    collision_row,
    collision_velocities,
    floor_crossing,
    floor_gap,
    max_energy_gain,
    native_fps,
    row_overlaps_floor,
)
from physverify.synth.perturbations._windows import UP, records_of, rest_mask
from physverify.synth.perturbations.r3_energy_gain import NAME as ENERGY_GAIN
from physverify.synth.perturbations.r3_energy_gain import first_ball_collision
from physverify.synth.plan import NONE, DatasetPlan, VariantPlan
from physverify.synth.settings import DatasetSettings
from physverify.synth.store import (
    CONTACTS_WORLD,
    FRAMES_DIR,
    GT_META,
    RECORD,
    SPEC,
    STATES_WORLD,
    VARIANT_STATES,
    VIDEO,
    DatasetStore,
    Step,
    read_json,
)

# Plugins that rewrite their target's rows with an analytic flight, which reads e_clean.
ANALYTIC_FLIGHT = frozenset(
    {"r3_restitution_gain", "r3_energy_gain", "r5a_gravity_switch", "r5a_object_gravity"}
)
RESTITUTION_GAIN = "r3_restitution_gain"
RERENDER_IF_CANDIDATES_MOVE = frozenset({"r4_resize", "r1_penetration", "r4_teleport"})
CONTACTS_FIXED = frozenset({NONE, "r5b_gravity_scale"})
LABEL_WATCH = ("r5a_", "r5b_")
MAGNITUDE_TOL = 1e-9
# Key of the velocity-convention migration in the maps below (its name carries the convention).
VELOCITY = "velocity"
REST_TOL_M = 1e-9
INTEGER_TOL = 1e-9
# The closed form assumes a ballistic step from the source row to k; Bullet's float64 velocities
# miss it by ~1e-8 m/s (|e_new - e_pred| <= 7.3e-9 on the dry run); WP-H itself moves e by ~1e-2.
CLOSED_FORM_TOL = 1e-6
FLOOR_TOL_M = 1e-9
# GT metadata a floor-crossing migration may move on a kept clip whose simulation has a
# penetrated bounce: R3a and e_clean are measured on it, and e_clean is a parameter of the edits.
FLOOR_META_PATHS = (
    "measures.R3a",
    "e_clean",
    "e_clean_above_one",
    "record.params.e_clean",
    "gt_version",
)
# GT metadata a pair-collision migration may move on a kept clip whose R3b the rule moves.
R3B_META_PATH = "measures.R3b"
PAIR_META_PATHS = (R3B_META_PATH, "record.magnitude", "gt_version")


@dataclass(frozen=True)
class AuditInputs:
    plan: DatasetPlan
    after: DatasetStore
    before: DatasetStore
    manifest: Path
    plugins: Mapping[str, Any]
    ds: DatasetSettings
    migration: str


@dataclass(frozen=True)
class Prediction:
    """Clips whose render the migration must discard, those it may discard, and the penetrated
    bounces behind each prediction."""

    certain: frozenset[str]
    possible: frozenset[str]
    bounces: Mapping[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass(frozen=True)
class BounceCase:
    """A sphere floor bounce: the row k the pre-WP-H ``floor_bounce`` resolved (its t_c = t_k
    fallback when z_k <= r), z_k - r there, and the (k, source row, t_c) it resolves to now."""

    k: int | None
    depth_m: float
    now: tuple[int, int, float] | None

    @property
    def penetrated(self) -> bool:
        return self.k is not None and self.depth_m <= 0.0

    @property
    def status(self) -> str:
        if self.now is None:
            return "none" if self.k is None else "unresolved"
        if self.now[0] != self.k:
            return "moved"
        return "previous_row" if self.now[1] != self.now[0] else "same_row"


def md5_of(paths: Iterable[Path]) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rendered_before(inputs: AuditInputs) -> list[VariantPlan]:
    return [v for v in inputs.plan.variants if inputs.before.is_done(Step.RENDER, v.clip_id)]


def velocity_prediction(inputs: AuditInputs) -> Prediction:
    """WP-G: every analytic flight moves (its velocities feed the flight); a rigid edit moves only
    when its candidate frames do."""
    rendered = rendered_before(inputs)
    return Prediction(
        frozenset(v.clip_id for v in rendered if v.perturbation in ANALYTIC_FLIGHT),
        frozenset(v.clip_id for v in rendered if v.perturbation in RERENDER_IF_CANDIDATES_MOVE),
    )


def floor_crossing_prediction(inputs: AuditInputs) -> Prediction:
    """WP-H: an R3a edit whose target's first floor bounce (``first_floor_impact``, the one it
    edits) was resolved on a penetrated row now starts its flight one row earlier; the other
    analytic flights with such a target only move if their e_clean changes a later bounce."""
    certain: set[str] = set()
    possible: set[str] = set()
    bounces: dict[str, list[dict[str, Any]]] = {}
    for variant in rendered_before(inputs):
        record = _record(inputs.before, variant)
        if variant.perturbation not in ANALYTIC_FLIGHT or record is None:
            continue
        scene, states, contacts = load_simulation(
            inputs.after, variant.sim_id, variant.sim_id, inputs.ds
        )
        cases = {
            o: first_bounce_case(states, contacts, states.object_index(o), gravity_of(scene))
            for o in record.target_objects
        }
        penetrated = {o: c for o, c in cases.items() if c is not None and c.penetrated}
        if not penetrated:
            continue
        first = next(iter(penetrated.values()))
        sure = variant.perturbation == RESTITUTION_GAIN and first.status == "previous_row"
        (certain if sure else possible).add(variant.clip_id)
        bounces[variant.clip_id] = [_bounce_row(o, c) for o, c in penetrated.items()]
    return Prediction(frozenset(certain), frozenset(possible), bounces)


def _bounce_row(object_id: str, case: BounceCase) -> dict[str, Any]:
    src, t_c = (None, math.nan) if case.now is None else case.now[1:]
    return {
        "object": object_id,
        "k": case.k,
        "src": src,
        "t_c": t_c,
        "depth_m": case.depth_m,
        "status": case.status,
    }


def covered_migrations(inputs: AuditInputs) -> list[str]:
    """What the run brought in: the velocity convention when it was that migration, plus the
    ground-truth fixes its manifest ``covers`` (a manifest from before the field covers itself)."""
    covers = read_json(inputs.manifest).get(COVERS_KEY)
    if inputs.migration == velocity_migration(inputs.ds):
        return [VELOCITY, *(covers or [])]
    return [str(name) for name in (covers if covers is not None else [inputs.migration])]


def prediction_for(inputs: AuditInputs) -> Prediction:
    """The union of the predictions of every migration the run covers."""
    parts = [PREDICTIONS[name](inputs) for name in covered_migrations(inputs)]
    certain = frozenset().union(*(p.certain for p in parts))
    possible = frozenset().union(*(p.possible for p in parts)) - certain
    bounces = {k: v for p in parts for k, v in p.bounces.items()}
    return Prediction(certain, possible, bounces)


def rerender_audit(inputs: AuditInputs, prediction: Prediction) -> dict[str, Any]:
    """A4' / H3: observed discarded renders against the prediction."""
    observed = set(rerender_clip_ids(inputs.manifest))
    return {
        "observed": sorted(observed),
        "predicted_certain": sorted(prediction.certain),
        "missing_from_certain": sorted(prediction.certain - observed),
        "predicted_possible": sorted(prediction.possible),
        "conditional_rerendered": {
            c: prediction.bounces.get(c, []) for c in sorted(observed & prediction.possible)
        },
        "unexpected": sorted(observed - prediction.certain - prediction.possible),
    }


def bounce_case(states: ObjectStateTable, index: int, t_guess: float, g: np.ndarray) -> BounceCase:
    radius = radius_of(states, index)
    gz = float(g[2])
    legacy = next(
        (
            k
            for k in bounce_rows(states, index, t_guess)
            if states.position[k, index, 2] <= radius
            or floor_crossing(states, index, k, radius, gz) is not None
        ),
        None,
    )
    depth = math.nan if legacy is None else float(states.position[legacy, index, 2]) - radius
    return BounceCase(legacy, depth, bounce_crossing(states, index, t_guess, g))


def first_bounce_case(
    states: ObjectStateTable, contacts: ContactTable, index: int, g: np.ndarray
) -> BounceCase | None:
    """The case of the bounce ``restitution_for_edit`` measures (None for a body that never
    bounces or is not a sphere)."""
    impact = first_floor_impact(str(states.object_ids[index]), contacts)
    if impact is None or not _sphere(states, index):
        return None
    return bounce_case(states, index, impact.t_start, g)


def integer_impact_check(inputs: AuditInputs, prediction: Prediction) -> dict[str, Any]:
    """H3 / C6: R3a edits whose impact frame is an integer (|f - round(f)| <= 1e-9, the mark of
    the old t_c = t_k fallback), before and after, against the certain re-renders."""

    def integral(store: DatasetStore) -> list[str]:
        out = []
        for variant in inputs.plan.variants:
            record = _record(store, variant)
            if variant.perturbation != RESTITUTION_GAIN or record is None:
                continue
            frame = float(record.params["impact_frame"])
            if abs(frame - round(frame)) <= INTEGER_TOL:
                out.append(variant.clip_id)
        return out

    before = integral(inputs.before)
    return {
        "before": before,
        "after": integral(inputs.after),
        "before_equals_certain": before == sorted(prediction.certain),
    }


def floor_audit(inputs: AuditInputs) -> dict[str, Any]:
    """H8' / C1: rows an analytic flight rewrote (from the first row that differs from the clean
    simulation), in the APPLY states and in the exported world states when present, must keep
    z - r >= -1e-9; clean simulations under the floor are only counted (Bullet's own impact
    penetration)."""
    checked, below = 0, {}
    for variant in inputs.plan.variants:
        record = _record(inputs.after, variant)
        clip_dir = inputs.after.clip_dir(variant.clip_id)
        paths = [p for p in (clip_dir / VARIANT_STATES, clip_dir / STATES_WORLD) if p.is_file()]
        if variant.perturbation not in ANALYTIC_FLIGHT or record is None or not paths:
            continue
        _, clean, _ = load_simulation(inputs.after, variant.sim_id, variant.clip_id, inputs.ds)
        checked += 1
        for path in paths:
            states = ObjectStateTable.load_npz(path)
            for object_id in record.target_objects:
                depth = _rewritten_depth(states, clean, states.object_index(object_id))
                if depth < -FLOOR_TOL_M:
                    below[f"{variant.clip_id}:{object_id}:{path.name}"] = depth
    clean_below = {}
    for sim_id in _simulated(inputs):
        _, states, _ = load_simulation(inputs.after, sim_id, sim_id, inputs.ds)
        depth = min((_min_depth(states, i, 0) for i in range(states.n_objects)), default=math.inf)
        if depth < -FLOOR_TOL_M:
            clean_below[sim_id] = depth
    return {
        "checked": checked,
        "rewritten_below_floor": below,
        "clean_sims_below_floor": len(clean_below),
        "deepest_clean_m": min(clean_below.values(), default=math.nan),
    }


def bounce_cases(inputs: AuditInputs) -> list[dict[str, Any]]:
    """C4 / C5: every sphere floor bounce of every simulation whose result moved or that sits on
    a non-ballistic neighbour, with old and new restitution and the closed form of the move."""
    rows: list[dict[str, Any]] = []
    for sim_id in _simulated(inputs):
        scene, states, contacts = load_simulation(inputs.after, sim_id, sim_id, inputs.ds)
        corrected = corrected_cells(states, contacts, scene)
        g = gravity_of(scene)
        for index in (i for i in range(states.n_objects) if _sphere(states, i)):
            object_id = str(states.object_ids[index])
            for impact in sorted(floor_impacts(object_id, contacts), key=lambda r: r.t_start):
                case = bounce_case(states, index, impact.t_start, g)
                row = _case_row(states, index, case, g, corrected)
                rows.append({"id": f"{sim_id}:{object_id}:{impact.t_start:.3f}", **row})
    return rows


def bounce_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """H12 / H13: penetrated bounces, closed-form mismatches and the C5 edge cases."""

    def ids(keep: Callable[[Mapping[str, Any]], bool]) -> list[str]:
        return [str(r["id"]) for r in rows if keep(r)]

    solved = [r for r in rows if r["status"] == "previous_row"]
    return {
        "bounces": len(rows),
        "penetrated": [r for r in rows if r["penetrated"]],
        "closed_form_tol": CLOSED_FORM_TOL,
        "closed_form_max_error": max(
            (abs(r["e_new"] - r["e_pred"]) for r in solved), default=math.nan
        ),
        "closed_form_mismatch": [
            str(r["id"]) for r in solved if not abs(r["e_new"] - r["e_pred"]) <= CLOSED_FORM_TOL
        ],
        "unresolved_now": ids(lambda r: r["status"] == "unresolved"),
        "row_moved": ids(lambda r: r["status"] == "moved"),
        "resting_previous_row": ids(lambda r: r["penetrated"] and r["resting_previous_row"]),
        "next_row_below_floor": ids(lambda r: r["next_row_below_floor"]),
    }


def gt_meta_audit(inputs: AuditInputs, explainers: Sequence[MetaExplainer]) -> dict[str, Any]:
    """H9 / I6: on kept clips every changed leaf of ``gt_meta.json`` must be explained by one of
    the migrations the run covers."""
    observed = set(rerender_clip_ids(inputs.manifest))
    compared, changed, unexplained = 0, {}, {}
    for variant in inputs.plan.variants:
        old, new = (_meta(s, variant) for s in (inputs.before, inputs.after))
        if old is None or new is None or variant.clip_id in observed:
            continue
        compared += 1
        paths = changed_paths(old, new)
        if not paths:
            continue
        changed[variant.clip_id] = paths
        loose = [p for p in paths if not any(explain(variant, p) for explain in explainers)]
        if loose:
            unexplained[variant.clip_id] = loose
    return {"compared": compared, "changed": changed, "unexplained": unexplained}


def floor_meta_explainer(rows: Sequence[Mapping[str, Any]]) -> MetaExplainer:
    """WP-H moves R3a and e_clean only where the simulation has a penetrated bounce."""
    penetrated = {str(r["id"]).split(":")[0] for r in rows if r["penetrated"]}

    def explains(variant: VariantPlan, path: str) -> bool:
        return _under(path, FLOOR_META_PATHS) and bool(
            {variant.scene_id, variant.sim_id} & penetrated
        )

    return explains


def e_clean_above_one_audit(
    inputs: AuditInputs, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """C4 b (WP-H): scene objects with e_clean above one before and after, the new ones with the
    depth of their penetrated row."""
    above: dict[str, dict[str, list[str]]] = {"before": {}, "after": {}}
    for variant in inputs.plan.variants:
        for side, store in (("before", inputs.before), ("after", inputs.after)):
            meta = _meta(store, variant)
            if meta is not None:
                above[side].setdefault(variant.scene_id, list(meta.get("e_clean_above_one", [])))
    before, after = (_scene_objects(above[side]) for side in ("before", "after"))
    depth = {
        f"{str(r['id']).split(':')[0]}:{str(r['id']).split(':')[1]}": r["depth_m"]
        for r in reversed(rows)
    }
    return {
        "before": before,
        "after": after,
        "new": {k: depth.get(k) for k in sorted(set(after) - set(before))},
        "removed": sorted(set(before) - set(after)),
    }


def changed_paths(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Dotted paths of the leaves that differ (NaN equals NaN; a missing leaf differs)."""
    a, b = _flat(old), _flat(new)
    return sorted(
        k for k in a.keys() | b.keys() if k not in a or k not in b or not _same(a[k], b[k])
    )


def media_audit(inputs: AuditInputs) -> dict[str, Any]:
    """A4': frames (and mp4) of the clips whose render was kept, before against after."""
    kept = [
        v.clip_id
        for v in inputs.plan.variants
        if inputs.before.is_done(Step.RENDER, v.clip_id)
        and v.clip_id not in set(rerender_clip_ids(inputs.manifest))
    ]
    frames: list[str] = []
    video: list[str] = []
    for clip_id in kept:
        old, new = inputs.before.clip_dir(clip_id), inputs.after.clip_dir(clip_id)
        if md5_of((old / FRAMES_DIR).glob("*.png")) != md5_of((new / FRAMES_DIR).glob("*.png")):
            frames.append(clip_id)
        if (old / VIDEO).is_file() and md5_of([old / VIDEO]) != md5_of([new / VIDEO]):
            video.append(clip_id)
    return {"kept": len(kept), "frames_differ": frames, "mp4_differ": video}


def contacts_audit(inputs: AuditInputs) -> dict[str, Any]:
    """A5: contact tables must not move on unedited and R5b clips; the others are listed."""
    fixed_changed: list[str] = []
    edited_changed: list[str] = []
    compared = 0
    for variant in inputs.plan.variants:
        old = inputs.before.clip_dir(variant.clip_id) / CONTACTS_WORLD
        new = inputs.after.clip_dir(variant.clip_id) / CONTACTS_WORLD
        if not (old.is_file() and new.is_file()):
            continue
        compared += 1
        if read_json(old) == read_json(new):
            continue
        target = fixed_changed if variant.perturbation in CONTACTS_FIXED else edited_changed
        target.append(variant.clip_id)
    return {"compared": compared, "fixed_changed": fixed_changed, "edited_changed": edited_changed}


def record_audit(inputs: AuditInputs) -> dict[str, Any]:
    """A6: the exported record equals ``final_record`` re-measured on the exported truth."""
    differ, checked = [], 0
    for variant in inputs.plan.variants:
        clip_dir = inputs.after.clip_dir(variant.clip_id)
        if not (clip_dir / GT_META).is_file():
            continue
        checked += 1
        record = final_record(
            inputs.plugins[variant.perturbation],
            PerturbationRecord.load_json(clip_dir / RECORD),
            variant,
            ObjectStateTable.load_npz(clip_dir / STATES_WORLD),
            ContactTable.load_json(clip_dir / CONTACTS_WORLD),
        )
        if read_json(clip_dir / GT_META)["record"] != record.to_dict():
            differ.append(variant.clip_id)
    return {"checked": checked, "differ": differ}


def label_audit(inputs: AuditInputs) -> dict[str, Any]:
    """A3 (labels) and the R5a/R5b label changes of A9: applied magnitude before and after."""
    changes: dict[str, Any] = {}
    off_nominal: dict[str, float] = {}
    for variant in inputs.plan.variants:
        after = _record(inputs.after, variant)
        if after is None:
            continue
        before = _record(inputs.before, variant)
        if variant.perturbation.startswith("r3_"):
            gap = abs(after.magnitude - variant.target_magnitude)
            if gap > MAGNITUDE_TOL:
                off_nominal[variant.clip_id] = gap
        watched = variant.perturbation.startswith(LABEL_WATCH)
        if watched and before is not None and before.magnitude != after.magnitude:
            changes[variant.clip_id] = {"before": before.magnitude, "after": after.magnitude}
    return {"r3_off_nominal": off_nominal, "r5_label_changes": changes}


def check_label_audit(inputs: AuditInputs, table: Path) -> dict[str, Any]:
    """A3: |R3 measured on the oracle - label| <= max(1e-3, 0.02 |label|) on the applied R3 clips
    the table covers."""
    rows = _table(table)
    covered = {clip_id for clip_id, _ in rows}
    within, outside = 0, {}
    for variant in inputs.plan.variants:
        record = _record(inputs.after, variant)
        if record is None or not variant.perturbation.startswith("r3_"):
            continue
        if variant.clip_id not in covered:
            continue
        for check_id in record.target_checks:
            value = rows.get((variant.clip_id, check_id))
            label = record.magnitude
            if value is not None and abs(value - label) <= max(1e-3, 0.02 * abs(label)):
                within += 1
            else:
                outside[f"{variant.clip_id}:{check_id}"] = {"measured": value, "label": label}
    return {"table": str(table), "within": within, "outside": outside}


def rest_audit(inputs: AuditInputs) -> dict[str, Any]:
    """A9: corrected cells at rest without a RESTING record (simulator velocity under v_tol, or
    the position still to both neighbours). Expected: a flight apex (v_z from + to -) and the
    release of a body dropped from rest (first frame, falling at the next)."""
    state = inputs.ds.model_copy(update={"velocity_convention": "state"})
    apex, release, other = 0, 0, []
    for sim_id in inputs.plan.sim_ids():
        if not inputs.after.is_done(Step.SIMULATE, sim_id):
            continue
        scene, states, clean = load_simulation(inputs.after, sim_id, sim_id, state)
        cells = corrected_cells(states, clean, scene)
        for k, i in zip(*np.nonzero(cells & _at_rest(states)), strict=True):
            if _apex(states, int(k), int(i)):
                apex += 1
            elif _release(states, int(k), int(i)):
                release += 1
            else:
                other.append(f"{sim_id}:{states.object_ids[i]}:{int(k)}")
    return {"apex_cells": apex, "release_cells": release, "other_cells": other}


def cache_audit(inputs: AuditInputs, cache_roots: Sequence[Path]) -> dict[str, Any]:
    """A10: cached stage metadata older than the frames of a re-rendered clip."""
    stale: dict[str, int] = {}
    for clip_id in rerender_clip_ids(inputs.manifest):
        frames = list((inputs.after.clip_dir(clip_id) / FRAMES_DIR).glob("*.png"))
        newest = min((p.stat().st_mtime for p in frames), default=float("inf"))
        for root in cache_roots:
            metas = StageCache(root).metas(DATASET_NAME, clip_id)
            count = sum(p.stat().st_mtime < newest for p in metas)
            if count:
                stale[f"{root}:{clip_id}"] = count
    return {"roots": [str(r) for r in cache_roots], "stale_meta": stale}


def audit(
    inputs: AuditInputs, cache_roots: Sequence[Path], tables: Sequence[Path]
) -> dict[str, Any]:
    prediction = prediction_for(inputs)
    report = {
        "migration": inputs.migration,
        "rerender": rerender_audit(inputs, prediction),
        "media": media_audit(inputs),
        "contacts": contacts_audit(inputs),
        "records": record_audit(inputs),
        "labels": label_audit(inputs),
        "check_labels": [check_label_audit(inputs, t) for t in tables],
        "rest": rest_audit(inputs),
        "cache": cache_audit(inputs, cache_roots),
    }
    extras = [EXTRA_AUDITS[name](inputs, prediction) for name in covered_migrations(inputs)]
    for extra in extras:
        report |= extra.report
    explainers = [e.explains for e in extras if e.explains is not None]
    if explainers:
        report["gt_meta"] = gt_meta_audit(inputs, explainers)
    return report


def floor_crossing_audits(inputs: AuditInputs, prediction: Prediction) -> Extra:
    rows = bounce_cases(inputs)
    report = {
        "impact_frames": integer_impact_check(inputs, prediction),
        "floor": floor_audit(inputs),
        "e_clean_above_one": e_clean_above_one_audit(inputs, rows),
        "bounces": bounce_summary(rows),
    }
    return Extra(report, floor_meta_explainer(rows))


def pair_collision_audits(inputs: AuditInputs, prediction: Prediction) -> Extra:
    tol = inputs.ds.contacts.tol_m
    rows = [
        r
        for v in _exported_variants(inputs.plan, inputs.after)
        for r in collision_rows(inputs.after, v, tol)
    ]
    r3b = r3b_audit(inputs)
    report = {
        "pair_rows": {
            "tol_m": tol,
            "touch_only": [r for r in rows if r["touch"] and not r["overlap"]],
            "legacy_only": [r for r in rows if r["legacy"] and not r["on_floor"]],
            "stops": [r for r in rows if not r["on_floor"] and r["reach_m"] >= r["gap_m"]],
        },
        "r3b": r3b,
        "energy_labels": energy_label_audit(inputs),
    }
    moved = set(r3b["predicted"])

    def explains(variant: VariantPlan, path: str) -> bool:
        edited = variant.perturbation == ENERGY_GAIN
        return variant.clip_id in moved and (
            path == R3B_META_PATH or (edited and path in PAIR_META_PATHS)
        )

    return Extra(report, explains)


def velocity_audits(inputs: AuditInputs, prediction: Prediction) -> Extra:
    return Extra({}, None)


type MetaExplainer = Callable[[VariantPlan, str], bool]


@dataclass(frozen=True)
class Extra:
    """The findings a migration adds to the report, and which kept-clip ``gt_meta.json`` changes
    it accounts for (None: the migration does not audit them)."""

    report: dict[str, Any]
    explains: MetaExplainer | None


type ExtraAudit = Callable[[AuditInputs, Prediction], Extra]


def _case_row(
    states: ObjectStateTable,
    index: int,
    case: BounceCase,
    g: np.ndarray,
    corrected: np.ndarray,
) -> dict[str, Any]:
    radius = radius_of(states, index)
    k = case.k if case.now is None else case.now[0]
    out: dict[str, Any] = {
        "status": case.status,
        "penetrated": case.penetrated,
        "k": case.k,
        "depth_m": case.depth_m,
        "resting_previous_row": False,
        "next_row_below_floor": k is not None and states.position[k + 1, index, 2] < radius,
    }
    if case.k is None or not case.penetrated:
        return out
    t, v = states.t, states.velocity[:, index]
    k = case.k
    v_plus_old = impact_velocity(v[k + 1], float(t[k + 1]), float(t[k]), g)
    out |= {
        "resting_previous_row": k >= 1 and not bool(corrected[k - 1, index]),
        "e_old": restitution(v[k], v_plus_old, UP),
        "e_new": math.nan,
        "e_pred": math.nan,
    }
    if case.now is None:
        return out
    _, src, t_c = case.now
    v_minus = impact_velocity(v[src], float(t[src]), t_c, g)
    v_plus = impact_velocity(v[k + 1], float(t[k + 1]), t_c, g)
    # Closed form of the move: both sides shifted by g (t_c - t_k) from the old t_c = t_k.
    shift = g * (t_c - float(t[k]))
    return out | {
        "src": src,
        "t_c": t_c,
        "e_new": restitution(v_minus, v_plus, UP),
        "e_pred": restitution(v[k] + shift, v_plus_old + shift, UP),
    }


def _rewritten_depth(exported: ObjectStateTable, clean: ObjectStateTable, index: int) -> float:
    """Least z - r over the rows from the first one that differs from the clean simulation (a
    row the export hid, NaN, differs from nothing)."""
    if not _sphere(exported, index):
        return math.inf
    mine = exported.position[:, index]
    differs = np.all(np.isfinite(mine), axis=1) & np.any(mine != clean.position[:, index], axis=1)
    rows = np.nonzero(differs)[0]
    return _min_depth(exported, index, int(rows[0])) if rows.size else math.inf


def _min_depth(states: ObjectStateTable, index: int, start: int) -> float:
    if not _sphere(states, index):
        return math.inf
    z = states.position[start:, index, 2]
    finite = z[np.isfinite(z)]
    return float(finite.min()) - radius_of(states, index) if finite.size else math.inf


def _sphere(states: ObjectStateTable, index: int) -> bool:
    return str(states.primitive[index]) == PrimitiveKind.SPHERE.value


def _simulated(inputs: AuditInputs) -> list[str]:
    return [s for s in inputs.plan.sim_ids() if inputs.after.is_done(Step.SIMULATE, s)]


def _meta(store: DatasetStore, variant: VariantPlan) -> dict[str, Any] | None:
    path = store.clip_dir(variant.clip_id) / GT_META
    return read_json(path) if path.is_file() else None


def _scene_objects(by_scene: Mapping[str, Sequence[str]]) -> list[str]:
    return sorted(f"{scene}:{o}" for scene, objects in by_scene.items() for o in objects)


def _flat(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        out |= _flat(value, f"{path}.") if isinstance(value, Mapping) else {path: value}
    return out


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


def _under(path: str, prefixes: Sequence[str]) -> bool:
    return any(path == p or path.startswith(f"{p}.") for p in prefixes)


def _record(store: DatasetStore, variant: VariantPlan) -> PerturbationRecord | None:
    path = store.clip_dir(variant.clip_id) / RECORD
    if store.is_skipped(variant.clip_id) or not path.is_file():
        return None
    return PerturbationRecord.load_json(path)


def _table(path: Path) -> dict[tuple[str, str], float | None]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["clip_id"], row["check_id"]): _number(row["magnitude"])
            for row in csv.DictReader(handle)
        }


def _number(cell: str) -> float | None:
    return None if cell in ("", "nan") else float(cell)


def _at_rest(states: ObjectStateTable) -> np.ndarray:
    slow = np.stack([rest_mask(states, i) for i in range(states.n_objects)], axis=1)
    step = np.linalg.norm(np.diff(states.position, axis=0), axis=2) <= REST_TOL_M
    still = np.zeros_like(slow)
    still[1:-1] = step[:-1] & step[1:]
    return slow | still


def _apex(states: ObjectStateTable, k: int, i: int) -> bool:
    if k == 0 or k == states.n_frames - 1:
        return False
    vz = states.velocity[:, i, 2]
    moving = np.linalg.norm(states.position[k + 1, i] - states.position[k - 1, i]) > REST_TOL_M
    return bool(moving and vz[k - 1] > 0.0 > vz[k + 1])


def _release(states: ObjectStateTable, k: int, i: int) -> bool:
    moving = np.linalg.norm(states.position[1, i] - states.position[0, i]) > REST_TOL_M
    return bool(k == 0 and moving and states.velocity[1, i, 2] < 0.0)


def pair_collision_prediction(inputs: AuditInputs) -> Prediction:
    """WP-I: an R3b edit moves only when the floor rule changes one of the four clean velocities
    its rescaling starts from; no other plugin reads them."""
    tol = inputs.ds.contacts.tol_m
    certain = set()
    for variant in rendered_before(inputs):
        if variant.perturbation != ENERGY_GAIN or _record(inputs.before, variant) is None:
            continue
        scene, states, contacts = load_simulation(
            inputs.after, variant.sim_id, variant.sim_id, inputs.ds
        )
        impact = first_ball_collision(states, contacts)
        if impact is None:
            continue
        t_c, g = impact.t_start / float(scene.frame_rate), gravity_of(scene)
        for object_id in (impact.object_a, impact.object_b):
            index = states.object_index(object_id)
            now = collision_velocities(states, contacts, index, t_c, g, tol)
            old = legacy_collision_velocities(states, contacts, index, t_c, g)
            if not all(np.array_equal(a, b) for a, b in zip(now, old, strict=True)):
                certain.add(variant.clip_id)
    return Prediction(frozenset(certain), frozenset())


def legacy_grounded(object_id: str, contacts: ContactTable, frame: int) -> bool:
    """The floor rule before WP-I: any floor record within one frame of the row."""
    return any(
        r.object_b == ENV_ID and math.floor(r.t_start) - 1 <= frame <= math.ceil(r.t_end) + 1
        for r in records_of(object_id, contacts)
    )


def legacy_collision_velocities(
    states: ObjectStateTable, contacts: ContactTable, index: int, t_c: float, g: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    k = collision_row(states, t_c)
    object_id = str(states.object_ids[index])
    out = []
    for row in (k, k + 1):
        v = states.velocity[row, index]
        grounded = legacy_grounded(object_id, contacts, row)
        out.append(v.copy() if grounded else impact_velocity(v, float(states.t[row]), t_c, g))
    return out[0], out[1]


def legacy_energy_gain(
    states: ObjectStateTable, contacts: ContactTable, masses: Mapping[str, float], g: np.ndarray
) -> float:
    fps = native_fps(states)
    values = []
    for r in _pair_impacts(contacts):
        (am, ap), (bm, bp) = (
            legacy_collision_velocities(
                states, contacts, states.object_index(o), r.t_start / fps, g
            )
            for o in (r.object_a, r.object_b)
        )
        values.append(kinetic_energy_gain(masses[r.object_a] / masses[r.object_b], am, bm, ap, bp))
    return max((v for v in values if math.isfinite(v)), default=math.nan)


def floor_reach(vz: float, dt: float, g: float) -> float:
    """Upper bound of the vertical distance a body covers between a row and t_c (``dt`` apart)
    in free fall: a flight row whose floor gap is within it may hide an unrecorded bounce."""
    return abs(vz) * abs(dt) + 0.5 * g * dt * dt


def collision_rows(store: DatasetStore, variant: VariantPlan, tol: float) -> list[dict[str, Any]]:
    """Rows k and k + 1 of both bodies of every ball-ball impact of an exported clip: the old
    flag, the record overlap, the floor gap, whether the new rule holds the row value and how far
    the body can fall before t_c (I11 stops on a flight row that could reach the floor)."""
    scene, states, contacts = _exported(store, variant)
    fps, g = native_fps(states), float(np.linalg.norm(gravity_of(scene)))
    rows = []
    for r in _pair_impacts(contacts):
        t_c = r.t_start / fps
        k = collision_row(states, t_c)
        for object_id in (r.object_a, r.object_b):
            index = states.object_index(object_id)
            for row in (k, k + 1):
                overlap = row_overlaps_floor(states, contacts, index, row, t_c)
                gap = floor_gap(states, index, row)
                vz = float(states.velocity[row, index, 2])
                rows.append(
                    {
                        "clip": variant.clip_id,
                        "object": object_id,
                        "impact": r.t_start,
                        "row": row,
                        "legacy": legacy_grounded(object_id, contacts, row),
                        "overlap": overlap,
                        "touch": gap <= tol,
                        "on_floor": overlap or gap <= tol,
                        "gap_m": gap,
                        "vz": vz,
                        "reach_m": floor_reach(vz, float(states.t[row]) - t_c, g),
                    }
                )
    return rows


def r3b_audit(inputs: AuditInputs) -> dict[str, Any]:
    """I6: ``measures.R3b`` before and after against the clips where the old and new rules differ
    on the truth exported before the migration (the old rule must reproduce the stored value)."""
    tol = inputs.ds.contacts.tol_m
    predicted, observed, legacy_differs = {}, {}, []
    for variant in _exported_variants(inputs.plan, inputs.before):
        scene, states, contacts = _exported(inputs.before, variant)
        masses = {o.object_id: o.mass for o in scene.objects}
        g = gravity_of(scene)
        old, new = (
            legacy_energy_gain(states, contacts, masses, g),
            max_energy_gain(states, contacts, masses, g, tol),
        )
        before = _meta(inputs.before, variant)
        after = _meta(inputs.after, variant)
        stored = math.nan if before is None else float(before["measures"]["R3b"])
        if not _close(old, stored):
            legacy_differs.append(variant.clip_id)
        if not _close(old, new):
            predicted[variant.clip_id] = {"before": old, "after": new}
        if after is not None and not _close(stored, float(after["measures"]["R3b"])):
            observed[variant.clip_id] = {"before": stored, "after": after["measures"]["R3b"]}
    return {
        "predicted": predicted,
        "observed": observed,
        "missing": sorted(set(predicted) - set(observed)),
        "unexpected": sorted(set(observed) - set(predicted)),
        "legacy_differs_from_stored": legacy_differs,
    }


def energy_label_audit(inputs: AuditInputs) -> dict[str, Any]:
    """I2: every R3b edit labelled at its nominal gain in ``record.json`` and, where exported, in
    ``gt_meta.json``; the clips without ``gt_meta.json`` are listed, not skipped."""
    checked, in_meta, without_meta, off = 0, 0, [], {}
    for variant in inputs.plan.variants:
        record, meta = _record(inputs.after, variant), _meta(inputs.after, variant)
        if variant.perturbation != ENERGY_GAIN or record is None:
            continue
        checked += 1
        labels = {"record": record.magnitude}
        if meta is None:
            without_meta.append(variant.clip_id)
        else:
            in_meta += 1
            labels["gt_meta"] = float(meta["record"]["magnitude"])
        gaps = {k: abs(v - variant.target_magnitude) for k, v in labels.items()}
        if not all(gap <= MAGNITUDE_TOL for gap in gaps.values()):
            off[variant.clip_id] = gaps
    return {
        "checked": checked,
        "checked_gt_meta": in_meta,
        "without_gt_meta": without_meta,
        "tol": MAGNITUDE_TOL,
        "off_nominal": off,
    }


def _pair_impacts(contacts: ContactTable) -> list[Any]:
    return [r for r in contacts.records if r.kind is ContactKind.IMPACT and r.object_b != ENV_ID]


def _exported_variants(plan: DatasetPlan, store: DatasetStore) -> list[VariantPlan]:
    return [v for v in plan.variants if (store.clip_dir(v.clip_id) / GT_META).is_file()]


def _exported(
    store: DatasetStore, variant: VariantPlan
) -> tuple[SceneSpec, ObjectStateTable, ContactTable]:
    clip_dir = store.clip_dir(variant.clip_id)
    return (
        SceneSpec.load_json(store.scene_dir(variant.sim_id) / SPEC),
        ObjectStateTable.load_npz(clip_dir / STATES_WORLD),
        ContactTable.load_json(clip_dir / CONTACTS_WORLD),
    )


def _close(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= MAGNITUDE_TOL


PREDICTIONS: Mapping[str, Callable[[AuditInputs], Prediction]] = {
    VELOCITY: velocity_prediction,
    FLOOR_CROSSING: floor_crossing_prediction,
    PAIR_COLLISION: pair_collision_prediction,
}
EXTRA_AUDITS: Mapping[str, ExtraAudit] = {
    VELOCITY: velocity_audits,
    FLOOR_CROSSING: floor_crossing_audits,
    PAIR_COLLISION: pair_collision_audits,
}
