"""Acceptance audit of a velocity migration (WP-G 1.C, A3-A10), against a copy taken before it.

Every finding is a list of clip ids (or cells) with the numbers that put it there, so the report
says what moved and why rather than only whether a threshold held."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import PerturbationRecord
from physverify.schema.states import ObjectStateTable
from physverify.synth.gt_export import DATASET_NAME, final_record
from physverify.synth.kinematics_gt import corrected_cells
from physverify.synth.migrate import rerender_clip_ids
from physverify.synth.perturb import load_simulation
from physverify.synth.perturbations._windows import rest_mask
from physverify.synth.plan import NONE, DatasetPlan, VariantPlan
from physverify.synth.settings import DatasetSettings
from physverify.synth.store import (
    CONTACTS_WORLD,
    FRAMES_DIR,
    GT_META,
    RECORD,
    STATES_WORLD,
    VIDEO,
    DatasetStore,
    Step,
    read_json,
)

RERENDER_ALWAYS = frozenset(
    {"r3_restitution_gain", "r3_energy_gain", "r5a_gravity_switch", "r5a_object_gravity"}
)
RERENDER_IF_CANDIDATES_MOVE = frozenset({"r4_resize", "r1_penetration", "r4_teleport"})
CONTACTS_FIXED = frozenset({NONE, "r5b_gravity_scale"})
LABEL_WATCH = ("r5a_", "r5b_")
MAGNITUDE_TOL = 1e-9
REST_TOL_M = 1e-9


@dataclass(frozen=True)
class AuditInputs:
    plan: DatasetPlan
    after: DatasetStore
    before: DatasetStore
    manifest: Path
    plugins: Mapping[str, Any]
    ds: DatasetSettings


def md5_of(paths: Iterable[Path]) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rerender_audit(inputs: AuditInputs) -> dict[str, Any]:
    """A4': observed discarded renders against the prediction, by perturbation."""
    observed = set(rerender_clip_ids(inputs.manifest))
    rendered = [v for v in inputs.plan.variants if inputs.before.is_done(Step.RENDER, v.clip_id)]
    always = [v.clip_id for v in rendered if v.perturbation in RERENDER_ALWAYS]
    maybe = {v.clip_id for v in rendered if v.perturbation in RERENDER_IF_CANDIDATES_MOVE}
    return {
        "observed": sorted(observed),
        "predicted_always": sorted(always),
        "missing_from_always": sorted(set(always) - observed),
        "conditional_rerendered": sorted(observed & maybe),
        "unexpected": sorted(observed - set(always) - maybe),
    }


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
            metas = (root / DATASET_NAME / clip_id).rglob("meta.json")
            count = sum(p.stat().st_mtime < newest for p in metas)
            if count:
                stale[f"{root}:{clip_id}"] = count
    return {"roots": [str(r) for r in cache_roots], "stale_meta": stale}


def audit(
    inputs: AuditInputs, cache_roots: Sequence[Path], tables: Sequence[Path]
) -> dict[str, Any]:
    return {
        "rerender": rerender_audit(inputs),
        "media": media_audit(inputs),
        "contacts": contacts_audit(inputs),
        "records": record_audit(inputs),
        "labels": label_audit(inputs),
        "check_labels": [check_label_audit(inputs, t) for t in tables],
        "rest": rest_audit(inputs),
        "cache": cache_audit(inputs, cache_roots),
    }


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
