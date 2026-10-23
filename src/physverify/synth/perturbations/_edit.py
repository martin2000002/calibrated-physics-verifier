"""Shared pieces of the trajectory perturbations: editing one object's columns of a state table,
provisional records, analytic floor-bounce records and simple in-frame / clearance tests."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.physics.geometry import Plane, Solid, matrix_from_quaternion, signed_distance
from physverify.physics.tracks import records_of
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable, ManifoldKind
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.layout import project, spec_intrinsics, spec_world_from_camera
from physverify.synth.perturbations._ballistics import Bounce, Flight
from physverify.synth.perturbations._measure import bounce_restitution, nominal_extents
from physverify.synth.perturbations._windows import UP, solid_at
from physverify.synth.settings import PerturbationCommon

Vec = NDArray[np.float64]
ANALYTIC_SOURCE = "analytic"


class PluginBase[P: PerturbationCommon]:
    """Constructor and family filter shared by every perturbation (``families`` empty = all)."""

    Params: ClassVar[type[PerturbationCommon]]
    families: ClassVar[tuple[str, ...]] = ()

    def __init__(self, params: BaseModel) -> None:
        if not isinstance(params, self.Params):
            raise TypeError(f"{type(self).__name__} expects {self.Params.__name__}")
        self.params: P = params  # type: ignore[assignment]

    def applicable(self, scene: SceneSpec) -> bool:
        return not self.families or scene.family in self.families


class Declared(Protocol):
    """What a plugin class declares (read by ``provisional_record``)."""

    name: ClassVar[str]
    target_checks: ClassVar[tuple[str, ...]]
    magnitude_unit: ClassVar[str]
    magnitude_kind: ClassVar[MagnitudeKind]


def provisional_record(
    plugin: Declared,
    magnitude: float,
    targets: Sequence[str],
    span: tuple[int, int] | None,
    params: dict[str, Any],
) -> PerturbationRecord:
    """Record with the target magnitude; ``perturb.stamp`` sets level, seed and the measured one."""
    return PerturbationRecord(
        name=plugin.name,
        target_checks=plugin.target_checks,
        magnitude=float(magnitude),
        magnitude_unit=plugin.magnitude_unit,
        magnitude_kind=plugin.magnitude_kind,
        level=0,
        target_objects=tuple(targets),
        frame_span=span,
        seed=0,
        params={"target_magnitude": float(magnitude), **params},
    )


def edit_object(
    states: ObjectStateTable, index: int, frames: slice | NDArray[np.int64], **columns: Any
) -> ObjectStateTable:
    """Copy of ``states`` with ``columns[name][...]`` written into ``name[frames, index]``."""
    changes: dict[str, Any] = {}
    for name, values in columns.items():
        array = np.array(getattr(states, name), copy=True)
        array[frames, index] = values
        changes[name] = array
    return dataclasses.replace(states, **changes)


def gravity_of(scene: SceneSpec) -> Vec:
    return np.asarray(scene.gravity, dtype=np.float64)


def spheres(states: ObjectStateTable) -> list[int]:
    return [
        i for i in range(states.n_objects) if str(states.primitive[i]) == PrimitiveKind.SPHERE.value
    ]


def radius_of(states: ObjectStateTable, index: int) -> float:
    return float(nominal_extents(states, index)[0]) / 2.0


def object_restitution(scene: SceneSpec, object_id: str) -> float:
    """The object's own coefficient; Bullet multiplies it by the floor's (1) [sin verificar]."""
    return next(o.restitution for o in scene.objects if o.object_id == object_id)


def bounce_record(object_id: str, bounce: Bounce, radius: float, fps: float) -> ContactRecord:
    """Floor impact of an analytically integrated sphere (point under the centre, normal +z)."""
    frame = bounce.t * fps
    return ContactRecord(
        object_a=object_id,
        object_b=ENV_ID,
        t_start=frame,
        t_end=frame,
        kind=ContactKind.IMPACT,
        normal=UP.copy(),
        points=(bounce.position - radius * UP).reshape(1, 3),
        manifold=ManifoldKind.POINT,
        confidence=1.0,
        source=ANALYTIC_SOURCE,
    )


def keep_records(
    contacts: ContactTable,
    keep: Callable[[ContactRecord], bool],
    extra: Iterable[ContactRecord] = (),
) -> ContactTable:
    records = tuple(r for r in contacts.records if keep(r)) + tuple(extra)
    ordered = tuple(sorted(records, key=lambda r: (r.t_start, r.object_a, r.object_b)))
    return dataclasses.replace(contacts, records=ordered)


def before(object_ids: Iterable[str], frame: float) -> Callable[[ContactRecord], bool]:
    """Keeps records of other objects, and records of ``object_ids`` that start before ``frame``."""
    ids = set(object_ids)

    def keep(record: ContactRecord) -> bool:
        involved = record.object_a in ids or record.object_b in ids
        return not involved or record.t_start < frame

    return keep


def camera_of(scene: SceneSpec) -> tuple[Vec, Vec]:
    return spec_intrinsics(scene.camera, scene.resolution), spec_world_from_camera(scene.camera)


def centres_in_image(scene: SceneSpec, positions: Vec) -> bool:
    """Every (finite) centre projects inside the image and in front of the camera."""
    K, pose = camera_of(scene)  # noqa: N806
    points = positions.reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    uv, z = project(points, K, pose)
    width, height = scene.resolution
    return bool(
        np.all(z > 0)
        and np.all((uv[:, 0] >= 0) & (uv[:, 0] <= width) & (uv[:, 1] >= 0) & (uv[:, 1] <= height))
    )


def clearance(
    states: ObjectStateTable,
    moved: Callable[[int], Solid],
    index: int,
    frames: Iterable[int],
    skip: Sequence[int] = (),
    with_floor: bool = False,
) -> float:
    """Smallest signed distance between the moved object and every other one (and the floor)."""
    best = math.inf
    for t in frames:
        solid = moved(t)
        for j in range(states.n_objects):
            if j != index and j not in skip:
                best = min(best, signed_distance(solid, solid_at(states, j, t)).distance)
        if with_floor:
            best = min(best, signed_distance(solid, Plane.ground()).distance)
    return best


def rotation_about(axis: Vec, angle: float) -> Vec:
    """Rodrigues rotation matrix about a unit ``axis``."""
    a = axis / np.linalg.norm(axis)
    skew = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    out: Vec = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew
    return out


def image_plane_tilt(scene: SceneSpec, magnitude: float, sign: float) -> Vec:
    """R(theta) about the horizontal projection of the optical axis, theta = 2 asin(m / 2), so
    |R g - g| / |g| = m and the tilt stays in the image plane."""
    forward = np.asarray(scene.camera.look_at, dtype=np.float64) - np.asarray(scene.camera.position)
    axis = forward - (forward @ UP) * UP
    return rotation_about(sign * axis / np.linalg.norm(axis), 2.0 * math.asin(magnitude / 2.0))


def shift_records(contacts: ContactTable, object_id: str, offset: Vec) -> ContactTable:
    """Records of ``object_id`` moved rigidly with it (provisional contacts of a moved body)."""

    def moved(record: ContactRecord) -> ContactRecord:
        if object_id not in (record.object_a, record.object_b):
            return record
        return dataclasses.replace(record, points=record.points + offset)

    return dataclasses.replace(contacts, records=tuple(moved(r) for r in contacts.records))


def half_width(solid: Solid, direction: Vec) -> float:
    """Half of the solid's extent along a unit ``direction``."""
    rotation = solid.pose.rotation
    half = solid.primitive.half
    if solid.primitive.shape == PrimitiveKind.SPHERE.value:
        return float(solid.primitive.radius)
    if solid.primitive.shape == PrimitiveKind.CYLINDER.value:
        along = abs(float(rotation[:, 2] @ direction))
        return (
            float(solid.primitive.radius) * math.sqrt(max(0.0, 1.0 - along**2))
            + float(half[2]) * along
        )
    return float(sum(abs(float(rotation[:, k] @ direction)) * float(half[k]) for k in range(3)))


def horizontal_face_normals(states: ObjectStateTable, index: int, frame: int) -> list[Vec]:
    """World directions of the box faces of object ``index`` that are (nearly) vertical."""
    rotation = matrix_from_quaternion(states.quaternion[frame, index])
    normals = [sign * rotation[:, k] for k in range(3) for sign in (1.0, -1.0)]
    return [n for n in normals if abs(float(n @ UP)) < 0.05]


def floor_impacts(object_id: str, contacts: ContactTable) -> list[ContactRecord]:
    """Every IMPACT record of the object with the floor (ENV on either side)."""
    return [
        r
        for r in records_of(object_id, contacts)
        if r.kind is ContactKind.IMPACT and ENV_ID in (r.object_a, r.object_b)
    ]


def first_floor_impact(object_id: str, contacts: ContactTable) -> ContactRecord | None:
    impacts = floor_impacts(object_id, contacts)
    return min(impacts, key=lambda r: r.t_start) if impacts else None


def restitution_for_edit(
    scene: SceneSpec, states: ObjectStateTable, contacts: ContactTable, index: int
) -> float:
    """Restitution measured on the first clean floor bounce (not clamped to 1: Bullet can return
    e > 1, ADR 0004 §9.1), else the object's own coefficient."""
    object_id = str(states.object_ids[index])
    impact = first_floor_impact(object_id, contacts)
    if impact is not None:
        measured = bounce_restitution(states, index, impact.t_start, gravity_of(scene))
        if math.isfinite(measured) and measured > 0.0:
            return measured
    return object_restitution(scene, object_id)


def with_flight(
    states: ObjectStateTable, index: int, start: int, flight: Flight
) -> ObjectStateTable:
    """Positions and velocities of frames ``start..T-1`` replaced by an analytic flight."""
    frames = slice(start, states.n_frames)
    return edit_object(states, index, frames, position=flight.positions, velocity=flight.velocities)


def pick(rng: np.random.Generator, items: Sequence[Any]) -> Any:
    return items[int(rng.integers(len(items)))]


def translated(states: ObjectStateTable, index: int, start: int, offset: Vec) -> ObjectStateTable:
    """Object ``index`` shifted rigidly by ``offset`` from frame ``start`` on (same velocity)."""
    frames = slice(start, states.n_frames)
    return edit_object(states, index, frames, position=states.position[frames, index] + offset)


def ballistic_state(states: ObjectStateTable, index: int, t: float, g: Vec) -> tuple[Vec, Vec]:
    """Position and velocity at time ``t`` (s), ballistic from the last frame <= t."""
    k = int(np.searchsorted(states.t, t, side="right")) - 1
    k = min(max(k, 0), states.n_frames - 1)
    tau = t - float(states.t[k])
    p, v = states.position[k, index], states.velocity[k, index]
    out_p: Vec = p + v * tau + 0.5 * g * tau**2
    out_v: Vec = v + g * tau
    return out_p, out_v


def object_mass(scene: SceneSpec, object_id: str) -> float:
    return next(o.mass for o in scene.objects if o.object_id == object_id)


def object_friction(scene: SceneSpec, object_id: str) -> float:
    return next(o.friction for o in scene.objects if o.object_id == object_id)
