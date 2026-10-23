"""Contact records of one canonical pair: the measured manifold of its rows, or the primitive
footprint of a face-to-face pair when Bullet's points cannot describe the patch."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.physics.contact_manifold import contact_manifold, reduce_manifold_indices
from physverify.physics.geometry import Body, Plane, Solid
from physverify.physics.signed_distance import signed_distance
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ManifoldKind
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.frame_windows import UP, solid_at
from physverify.synth.settings import ContactSettings

Vec = NDArray[np.float64]
CLOSEST_SOURCE = "pybullet_closest"
PRIMITIVE_SOURCE = "primitive"
UPRIGHT_COS = math.cos(math.radians(10.0))


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


def frame_of(states: ObjectStateTable, frame: float) -> int:
    return min(states.n_frames - 1, max(0, math.floor(float(frame) + 1e-9)))


def finite(body: Body) -> bool:
    if not isinstance(body, Solid):
        return True
    return bool(np.isfinite(body.pose.position).all() and np.isfinite(body.pose.rotation).all())


def mean_normal(normals: Vec) -> Vec:
    total = normals.sum(axis=0)
    norm = float(np.linalg.norm(total))
    return total / norm if norm > 1e-12 else normals[0] / np.linalg.norm(normals[0])


def to_record(
    pair: PairSamples,
    kind: ContactKind,
    states: ObjectStateTable,
    cfg: ContactSettings,
    source: str,
) -> ContactRecord:
    normal = mean_normal(pair.normal)
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
    for frame in sorted({frame_of(states, f) for f in pair.frame}):
        body_a, body_b = body(states, pair.a, frame), body(states, pair.b, frame)
        if finite(body_a) and finite(body_b):
            depths.append(-signed_distance(body_a, body_b).distance)
    return max(depths)


def _primitive_record(
    pair: PairSamples, states: ObjectStateTable, cfg: ContactSettings, normal: Vec, reason: str
) -> ContactRecord | None:
    """``contact_manifold`` of a face-to-face pair at the middle frame of the run; None for other
    pairs or when the primitives do not touch within ``max(tol_m, 1e-4)``."""
    mid = round(float(pair.frame[0] + pair.frame[-1]) / 2.0)
    body_a, body_b = body(states, pair.a, mid), body(states, pair.b, mid)
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


def body(states: ObjectStateTable, object_id: str, frame: int) -> Body:
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
