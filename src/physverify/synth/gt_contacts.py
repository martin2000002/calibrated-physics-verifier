"""Ground-truth contact tables: PyBullet collision events of a simulated clip, or getClosestPoints
samples of an edited trajectory plus the analytic impacts of its provisional table.

Normals point from b to a and the pair is canonical: ``object_a`` is never the environment and
two movable objects are ordered by id. Times are fractional native frames clipped to [0, T - 1].
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from physverify.physics.contact_manifold import (
    contact_manifold,
    convex_hull_indices,
    polygon_area,
    reduce_manifold_indices,
)
from physverify.physics.geometry import Body, Manifold, Plane, Solid, project_to_plane
from physverify.physics.signed_distance import signed_distance
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import (
    ContactKind,
    ContactRecord,
    ContactTable,
    ManifoldKind,
    RawCollisions,
)
from physverify.schema.scene import SceneSpec
from physverify.schema.states import (
    CoordFrame,
    LengthUnit,
    ObjectStateTable,
    PrimitiveKind,
    ground_truth_source_key,
)
from physverify.synth.perturbations._frame_windows import UP, solid_at
from physverify.synth.settings import ContactSettings

Vec = NDArray[np.float64]
EVENT_STEP_OFFSET = -1.5
EVENTS_SOURCE = "pybullet_events"
CLOSEST_SOURCE = "pybullet_closest"
PRIMITIVE_SOURCE = "primitive"
UPRIGHT_COS = math.cos(math.radians(10.0))
_TINY_AREA = 1e-8


@dataclass(frozen=True)
class ClosestSamples:
    """``getClosestPoints`` rows of the ``contacts`` task (one per point, pair and frame)."""

    frame: NDArray[np.int64]
    a: NDArray[np.str_]
    b: NDArray[np.str_]
    point_a: Vec
    point_b: Vec
    normal_on_b: Vec
    distance: Vec

    @classmethod
    def load(cls, path: Path) -> ClosestSamples:
        with np.load(path, allow_pickle=False) as data:
            return cls(**{f.name: np.asarray(data[f.name]) for f in fields(cls)})


@dataclass(frozen=True)
class PairSamples:
    """Rows of one canonical pair sorted by frame; NaN distance or force when not measured."""

    a: str
    b: str
    frame: Vec
    point: Vec
    normal: Vec
    distance: Vec
    force: Vec


def event_frames_to_native(
    frames: NDArray[np.float64], settle_frames: int, *, steps_per_frame: int
) -> Vec:
    """Kubric reads contacts before each step, so ``frame = k / spf`` reports the state of step
    k - 1 and the touch happened during step k - 2 -> k - 1: the native (post-settle) frame is
    ``frame - settle + EVENT_STEP_OFFSET / spf`` with the offset at the middle of that step
    (measured +1.2 and +1.5 steps late on two floor impacts, ADR 0004 Implementation)."""
    out: Vec = (
        np.asarray(frames, dtype=np.float64) - settle_frames + EVENT_STEP_OFFSET / steps_per_frame
    )
    return out


def split_intervals(frames: Vec, gap_frames: float) -> list[slice]:
    """Runs of sorted ``frames`` with no gap larger than ``gap_frames``."""
    if frames.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(frames) > gap_frames) + 1
    bounds = [0, *cuts.tolist(), int(frames.size)]
    return [slice(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)]


def classify_interval(duration_frames: float, cfg: ContactSettings) -> ContactKind:
    return ContactKind.IMPACT if duration_frames < cfg.impact_max_frames else ContactKind.RESTING


@dataclass(frozen=True)
class Patch:
    """Normal (b to a), area and centroid of one frame of a resting contact."""

    normal: Vec
    area: float
    centroid: Vec


def split_runs(patches: Sequence[Patch], cfg: ContactSettings) -> list[range]:
    """Consecutive patches grouped into runs: a run ends when a patch turns by more than
    ``normal_split_deg``, changes its area by more than ``area_split_rel`` or moves its centroid
    by more than ``drift_split_m`` with respect to the first patch of the run."""
    runs: list[range] = []
    start = 0
    for k in range(1, len(patches)):
        if _differs(patches[start], patches[k], cfg):
            runs.append(range(start, k))
            start = k
    if patches:
        runs.append(range(start, len(patches)))
    return runs


def resting_runs(pair: PairSamples, states: ObjectStateTable, cfg: ContactSettings) -> list[slice]:
    """Row slices of one resting interval, split where the per-frame patch changes."""
    groups = _frame_groups(pair.frame)
    patches = [frame_patch(pair, group, states, cfg) for group in groups]
    return [
        slice(groups[run.start].start, groups[run.stop - 1].stop)
        for run in split_runs(patches, cfg)
    ]


def frame_patch(
    pair: PairSamples, rows: slice, states: ObjectStateTable, cfg: ContactSettings
) -> Patch:
    """Patch of the two primitives at the rows' frame, so that the split does not depend on how
    many points Bullet sampled (one per frame from ``getClosestPoints``, two to four per step
    from the events); the sampled rows when the primitives do not touch within ``tol_m``."""
    manifold = _pose_manifold(
        states, pair.a, pair.b, _frame_of(states, pair.frame[rows.start]), cfg
    )
    if manifold is None:
        return _patch(pair.point[rows], pair.normal[rows])
    return _patch(manifold.points, manifold.normals)


def _frame_of(states: ObjectStateTable, frame: float) -> int:
    return min(states.n_frames - 1, max(0, math.floor(float(frame) + 1e-9)))


def _pose_manifold(
    states: ObjectStateTable, a: str, b: str, frame: int, cfg: ContactSettings
) -> Manifold | None:
    body_a, body_b = _body(states, a, frame), _body(states, b, frame)
    if not (isinstance(body_a, Solid) and _finite(body_a) and _finite(body_b)):
        return None
    return contact_manifold(body_a, body_b, max(cfg.tol_m, 1e-4))


def _finite(body: Body) -> bool:
    if not isinstance(body, Solid):
        return True
    return bool(np.isfinite(body.pose.position).all() and np.isfinite(body.pose.rotation).all())


def _frame_groups(frames: Vec) -> list[slice]:
    keys = np.floor(frames + 1e-9)
    cuts = np.flatnonzero(np.diff(keys) != 0) + 1
    bounds = [0, *cuts.tolist(), int(frames.size)] if frames.size else []
    return [slice(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)]


def _patch(points: Vec, normals: Vec) -> Patch:
    normal = _mean_normal(normals)
    area = polygon_area(points, normal) if points.shape[0] >= 3 else 0.0
    return Patch(
        normal, area, _centroid(points, normal) if area > _TINY_AREA else points.mean(axis=0)
    )


def _centroid(points: Vec, normal: Vec) -> Vec:
    """Area centroid of the convex patch (the mean of its vertices depends on which of them the
    reduction kept)."""
    hull = points[convex_hull_indices(project_to_plane(points, normal))]
    fan = [(hull[0], hull[k], hull[k + 1]) for k in range(1, hull.shape[0] - 1)]
    weights = np.array([float(np.linalg.norm(np.cross(b - a, c - a))) for a, b, c in fan])
    centres = np.array([(a + b + c) / 3.0 for a, b, c in fan])
    out: Vec = weights @ centres / weights.sum()
    return out


def _differs(ref: Patch, patch: Patch, cfg: ContactSettings) -> bool:
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(ref.normal @ patch.normal)))))
    largest = max(ref.area, patch.area)
    area_jump = largest > _TINY_AREA and abs(ref.area - patch.area) / largest > cfg.area_split_rel
    drift = float(np.linalg.norm(ref.centroid - patch.centroid)) > cfg.drift_split_m
    return angle > cfg.normal_split_deg or area_jump or drift


def _mean_normal(normals: Vec) -> Vec:
    total = normals.sum(axis=0)
    norm = float(np.linalg.norm(total))
    return total / norm if norm > 1e-12 else normals[0] / np.linalg.norm(normals[0])


def from_raw_events(
    raw: RawCollisions, scene: SceneSpec, states: ObjectStateTable, cfg: ContactSettings
) -> ContactTable:
    spf = scene.step_rate // scene.frame_rate
    native = event_frames_to_native(raw.frame, scene.settle_frames, steps_per_frame=spf)
    keep = native >= 0.0
    nan = np.full(int(keep.sum()), np.nan)
    pairs = canonical_pairs(
        states,
        native[keep],
        raw.a[keep].astype(str),
        raw.b[keep].astype(str),
        raw.position[keep],
        raw.normal[keep],
        nan,
        raw.force[keep],
    )
    records = [r for pair in pairs for r in _event_records(pair, states, cfg)]
    return contact_table(states.clip_id, records)


def _event_records(
    pair: PairSamples, states: ObjectStateTable, cfg: ContactSettings
) -> list[ContactRecord]:
    out: list[ContactRecord] = []
    for interval in split_intervals(pair.frame, cfg.gap_frames):
        rows = _rows(pair, interval)
        duration = float(rows.frame[-1] - rows.frame[0])
        if classify_interval(duration, cfg) is ContactKind.IMPACT:
            out.append(to_record(rows, ContactKind.IMPACT, states, cfg, EVENTS_SOURCE))
            continue
        if approach_speed(states, rows) > cfg.impact_min_speed:
            opening = rows.frame < rows.frame[0] + cfg.impact_max_frames
            out.append(
                to_record(_mask(rows, opening), ContactKind.IMPACT, states, cfg, EVENTS_SOURCE)
            )
        out.extend(_resting(rows, states, cfg, EVENTS_SOURCE))
    return out


def from_closest(
    samples: ClosestSamples,
    provisional: ContactTable,
    states: ObjectStateTable,
    cfg: ContactSettings,
) -> ContactTable:
    """Resting contacts from closest points within ``tol_m`` (runs of consecutive frames) plus the
    impacts of the provisional table; single-frame runs next to an impact of the pair are the
    impact itself and are dropped."""
    keep = samples.distance <= cfg.tol_m
    pairs = canonical_pairs(
        states,
        samples.frame[keep].astype(np.float64),
        samples.a[keep].astype(str),
        samples.b[keep].astype(str),
        (samples.point_a[keep] + samples.point_b[keep]) / 2.0,
        samples.normal_on_b[keep],
        samples.distance[keep],
        np.full(int(keep.sum()), np.nan),
    )
    impacts = [r for r in provisional.records if r.kind is ContactKind.IMPACT]
    records: list[ContactRecord] = list(impacts)
    for pair in pairs:
        for interval in split_intervals(pair.frame, 1.0 + 1e-9):
            for record in _resting(_rows(pair, interval), states, cfg, CLOSEST_SOURCE):
                if not _touches_impact(record, impacts):
                    records.append(record)
    return contact_table(states.clip_id, records)


def _touches_impact(record: ContactRecord, impacts: list[ContactRecord]) -> bool:
    if record.t_end > record.t_start:
        return False
    return any(
        {i.object_a, i.object_b} == {record.object_a, record.object_b}
        and i.t_start - 1.0 <= record.t_start <= i.t_end + 1.0
        for i in impacts
    )


def contact_table(clip_id: str, records: list[ContactRecord]) -> ContactTable:
    ordered = tuple(
        sorted(records, key=lambda r: (r.t_start, r.object_a, r.object_b, r.kind.value))
    )
    return ContactTable(
        clip_id=clip_id,
        frame=CoordFrame.WORLD_ZUP,
        unit=LengthUnit.METER,
        source_key=ground_truth_source_key(clip_id),
        records=ordered,
    )


def canonical_pairs(
    states: ObjectStateTable,
    frame: Vec,
    a: NDArray[np.str_],
    b: NDArray[np.str_],
    point: Vec,
    normal: Vec,
    distance: Vec,
    force: Vec,
) -> list[PairSamples]:
    """Rows grouped by canonical pair, normals flipped to point from b to a (checked against the
    centre of a, because the order of Kubric's ``instances`` is not trusted)."""
    ids = states.object_ids.astype(str).tolist()
    last = states.n_frames - 1
    keys: list[tuple[str, str]] = []
    normals = np.array(normal, dtype=np.float64, copy=True).reshape(-1, 3)
    for k in range(frame.size):
        first, second = str(a[k]), str(b[k])
        if first == ENV_ID or (second != ENV_ID and second < first):
            first, second = second, first
            normals[k] = -normals[k]
        centre = states.position[min(last, max(0, round(float(frame[k])))), ids.index(first)]
        if float(normals[k] @ (centre - point[k])) < 0.0:
            normals[k] = -normals[k]
        keys.append((first, second))
    out: list[PairSamples] = []
    for key in sorted(set(keys)):
        rows = np.array([k for k, pair in enumerate(keys) if pair == key], dtype=np.int64)
        rows = rows[np.argsort(frame[rows], kind="stable")]
        clipped = np.clip(frame[rows], 0.0, float(last))
        out.append(
            PairSamples(
                key[0], key[1], clipped, point[rows], normals[rows], distance[rows], force[rows]
            )
        )
    return out


def _rows(pair: PairSamples, rows: slice) -> PairSamples:
    return PairSamples(
        pair.a,
        pair.b,
        pair.frame[rows],
        pair.point[rows],
        pair.normal[rows],
        pair.distance[rows],
        pair.force[rows],
    )


def _mask(pair: PairSamples, mask: NDArray[np.bool_]) -> PairSamples:
    return PairSamples(
        pair.a,
        pair.b,
        pair.frame[mask],
        pair.point[mask],
        pair.normal[mask],
        pair.distance[mask],
        pair.force[mask],
    )


def _resting(
    pair: PairSamples, states: ObjectStateTable, cfg: ContactSettings, source: str
) -> list[ContactRecord]:
    return [
        to_record(_rows(pair, run), ContactKind.RESTING, states, cfg, source)
        for run in resting_runs(pair, states, cfg)
    ]


def approach_speed(states: ObjectStateTable, pair: PairSamples) -> float:
    """-(v_a - v_b) . n at the last frame before the interval starts (v_b = 0 for the floor)."""
    ids = states.object_ids.astype(str).tolist()
    frame = min(states.n_frames - 1, max(0, math.floor(float(pair.frame[0]))))
    v_a = states.velocity[frame, ids.index(pair.a)]
    v_b = np.zeros(3) if pair.b == ENV_ID else states.velocity[frame, ids.index(pair.b)]
    speed = -float((v_a - v_b) @ _mean_normal(pair.normal))
    return speed if math.isfinite(speed) else 0.0


def to_record(
    pair: PairSamples,
    kind: ContactKind,
    states: ObjectStateTable,
    cfg: ContactSettings,
    source: str,
) -> ContactRecord:
    normal = _mean_normal(pair.normal)
    t_start, t_end = float(pair.frame[0]), float(pair.frame[-1])
    forces = pair.force[np.isfinite(pair.force)]
    peak = float(forces.max()) if forces.size else None
    if kind is ContactKind.IMPACT:
        strongest = int(np.nanargmax(pair.force)) if forces.size else 0
        return ContactRecord(
            object_a=pair.a,
            object_b=pair.b,
            t_start=t_start,
            t_end=t_end,
            kind=kind,
            normal=normal,
            points=pair.point[strongest].reshape(1, 3),
            manifold=ManifoldKind.POINT,
            peak_force=peak,
            confidence=1.0,
            source=source,
        )
    indices = reduce_manifold_indices(pair.point, normal)
    reason = _primitive_reason(source, int(indices.size))
    if reason is not None:
        primitive = _primitive_record(pair, states, cfg, normal, reason)
        if primitive is not None:
            return primitive
    distances = pair.distance[indices]
    measured = bool(np.all(np.isfinite(distances)))
    return ContactRecord(
        object_a=pair.a,
        object_b=pair.b,
        t_start=t_start,
        t_end=t_end,
        kind=kind,
        normal=normal,
        points=pair.point[indices],
        manifold=ManifoldKind.MEASURED,
        point_normals=pair.normal[indices],
        point_distances=distances if measured else None,
        max_penetration=max_penetration(pair, states),
        peak_force=peak,
        confidence=1.0,
        source=source,
    )


def _primitive_reason(source: str, n_points: int) -> str | None:
    """Why a resting record takes the primitive footprint, or None to keep Bullet's points.

    ``getClosestPoints`` gives about one point per pair and frame, so the union over a run
    depends on how many frames it lasted, not on the geometry: face-to-face pairs from closest
    points always take the primitive. Event rows (two to four points per step) only when they
    left fewer than 3 points."""
    if source == CLOSEST_SOURCE:
        return source
    return f"sparse_{source}" if n_points < 3 else None


def max_penetration(pair: PairSamples, states: ObjectStateTable) -> float:
    """max(0, -min distance) of the measured rows; on the primitives when a distance is unknown."""
    if pair.distance.size and bool(np.all(np.isfinite(pair.distance))):
        return max(0.0, -float(pair.distance.min()))
    return primitive_penetration(pair, states)


def primitive_penetration(pair: PairSamples, states: ObjectStateTable) -> float:
    """Deepest overlap of the two primitives over the rows' frames, for event rows that carry no
    distance (Kubric's collision events have none, ADR 0004 §9.1)."""
    depths = [0.0]
    for frame in sorted({_frame_of(states, f) for f in pair.frame}):
        body_a, body_b = _body(states, pair.a, frame), _body(states, pair.b, frame)
        if _finite(body_a) and _finite(body_b):
            depths.append(-signed_distance(body_a, body_b).distance)
    return max(depths)


def _primitive_record(
    pair: PairSamples, states: ObjectStateTable, cfg: ContactSettings, normal: Vec, reason: str
) -> ContactRecord | None:
    """``contact_manifold`` of a face-to-face pair at the middle frame of the run; None for other
    pairs or when the primitives do not touch within ``max(tol_m, 1e-4)``."""
    mid = round(float(pair.frame[0] + pair.frame[-1]) / 2.0)
    body_a, body_b = _body(states, pair.a, mid), _body(states, pair.b, mid)
    if not (isinstance(body_a, Solid) and _face_to_face(body_a, body_b)):
        return None
    manifold = contact_manifold(body_a, body_b, max(cfg.tol_m, 1e-4))
    if manifold is None:
        return None
    keep = reduce_manifold_indices(manifold.points, normal)
    return ContactRecord(
        object_a=pair.a,
        object_b=pair.b,
        t_start=float(pair.frame[0]),
        t_end=float(pair.frame[-1]),
        kind=ContactKind.RESTING,
        normal=normal,
        points=manifold.points[keep],
        manifold=ManifoldKind.PRIMITIVE,
        point_normals=manifold.normals[keep],
        point_distances=manifold.distances[keep],
        max_penetration=max_penetration(pair, states),
        confidence=1.0,
        source=f"{PRIMITIVE_SOURCE}:{reason}",
    )


def _body(states: ObjectStateTable, object_id: str, frame: int) -> Body:
    if object_id == ENV_ID:
        return Plane.ground()
    return solid_at(states, states.object_index(object_id), frame)


def _face_to_face(a: Solid, b: Body) -> bool:
    """Box-box, box-floor and upright cylinder pairs, where a patch (not a point) is expected."""
    return _flat(a) and (not isinstance(b, Solid) or _flat(b))


def _flat(solid: Solid) -> bool:
    if solid.primitive.shape == PrimitiveKind.BOX.value:
        return True
    if solid.primitive.shape == PrimitiveKind.CYLINDER.value:
        return abs(float(solid.pose.rotation[:, 2] @ UP)) >= UPRIGHT_COS
    return False
