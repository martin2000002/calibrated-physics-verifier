"""Analytic scenes for the check tests.

Scenes are built in a z-up world in meters and stored as a camera sees them (camera_opencv), so
a check can only learn "down" from ``gravity_dir``. Tables always hold t = frame_index * DT.
Dynamic scenes have FRAMES native frames with the event near the middle, so a table strided
with s = 2 or 3 keeps >= 4 rows on each side of it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from physverify.checks._base import CheckInputs
from physverify.extraction.matching import take_frames
from physverify.extraction.providers._common import centred_intrinsics
from physverify.physics.contact_manifold import contact_manifold
from physverify.physics.geometry import Body, Plane, Pose, Primitive, Solid, quaternion_from_matrix
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, SamplingSpec, Split, TimeSource, resolve_sampling
from physverify.schema.constants import ENV_ID, STANDARD_GRAVITY
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable, ManifoldKind
from physverify.schema.perception import CameraModel, IntrinsicsSource
from physverify.schema.scale import EffectiveScale, ScaleSource
from physverify.schema.scene import SceneSpec
from physverify.schema.stages import STAGE_ORDER
from physverify.schema.states import (
    CoordFrame,
    LengthUnit,
    ObjectStateTable,
    PrimitiveKind,
    ground_truth_source_key,
)

FloatArray = NDArray[np.float64]

FPS = 24.0
DT = 1.0 / FPS
FRAMES = 32
EVENT_FRAME = 15.4
CONTACT_TOL = 1e-6
UP = np.array([0.0, 0.0, 1.0])
DOWN = -UP
GROUND = Plane.ground()
CAMERA_ORIGIN = np.array([0.0, -4.0, 1.0])
BALL_RADIUS = 0.1
FOCAL_PX = 500.0
"""Focal length of the scenario camera, in processing pixels (R4_jump v2 reads its pixel floor)."""
DATASET = "scenario"


@dataclass(frozen=True, eq=False)
class View:
    """Camera with ``rotation`` = camera_from_world and centre ``origin`` in the world."""

    rotation: FloatArray
    origin: FloatArray

    def point(self, p: Any) -> FloatArray:
        return np.asarray((np.asarray(p, dtype=np.float64) - self.origin) @ self.rotation.T)

    def vector(self, v: Any) -> FloatArray:
        return np.asarray(np.asarray(v, dtype=np.float64) @ self.rotation.T)


def camera(pitch_deg: float = 0.0) -> View:
    """Camera looking along world +y, tilted down by ``pitch_deg`` (x right, y down, z forward)."""
    pitch = math.radians(pitch_deg)
    right = np.array([1.0, 0.0, 0.0])
    forward = np.array([0.0, math.cos(pitch), -math.sin(pitch)])
    world_from_camera = np.column_stack([right, np.cross(forward, right), forward])
    return View(rotation=world_from_camera.T, origin=CAMERA_ORIGIN)


@dataclass(frozen=True, eq=False)
class SceneObject:
    """One object in world coordinates over the native frames of the scene."""

    object_id: str
    kind: PrimitiveKind
    extents: FloatArray  # (3,) or (T, 3) full extents along the body axes
    position: FloatArray  # (T, 3)
    velocity: FloatArray  # (T, 3)
    acceleration: FloatArray  # (T, 3)
    rotation: FloatArray = field(default_factory=lambda: np.eye(3))  # world_from_body
    label: str = "object"

    @property
    def frames(self) -> int:
        return int(self.position.shape[0])

    def solid_at(self, position: Any) -> Solid:
        extents = np.asarray(self.extents, dtype=np.float64).reshape(-1, 3)[0]
        return Solid(Primitive(self.kind, extents), Pose(np.asarray(position), self.rotation))


@dataclass(frozen=True, eq=False)
class Case:
    """A scene ready for the checks, plus the camera it was expressed in."""

    clip: ClipManifest
    states: ObjectStateTable
    contacts: ContactTable | None
    scale: EffectiveScale | None
    view: View
    scene: SceneSpec | None = None

    def inputs(
        self,
        level: InputLevel = InputLevel.ORACLE,
        provenance: Mapping[str, str] | None = None,
    ) -> CheckInputs:
        source = "none" if self.scale is None else self.scale.source.value
        return CheckInputs(
            clip=self.clip,
            states=self.states,
            contacts=self.contacts,
            scale=self.scale,
            input_level=level,
            provenance=dict(provenance or {"scale.source": source}),
            scene=self.scene,
            camera=pinhole(self.clip),
        )


def pinhole(clip: ClipManifest, focal: float = FOCAL_PX, pose: Any = None) -> CameraModel:
    """Centred pinhole camera of ``focal`` pixels at the clip's native size."""
    sampling = resolve_sampling(clip, SamplingSpec())
    return CameraModel(
        sampling=sampling,
        K=centred_intrinsics(focal, sampling),
        source=IntrinsicsSource.ASSUMED_FOV,
        fov_deg=60.0,
        fov_sigma_deg=0.0,
        world_from_camera=pose,
    )


# Building blocks.


def times(frames: int = FRAMES) -> FloatArray:
    return np.arange(frames, dtype=np.float64) * DT


def yaw(degrees: float) -> FloatArray:
    """Rotation about world z (the gravity axis)."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def cube(side: float) -> FloatArray:
    return np.full(3, side)


def still(
    object_id: str,
    kind: PrimitiveKind,
    extents: Any,
    centre: Any,
    frames: int = FRAMES,
    rotation: Any = None,
) -> SceneObject:
    return SceneObject(
        object_id=object_id,
        kind=kind,
        extents=np.asarray(extents, dtype=np.float64),
        position=np.tile(np.asarray(centre, dtype=np.float64), (frames, 1)),
        velocity=np.zeros((frames, 3)),
        acceleration=np.zeros((frames, 3)),
        rotation=np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64),
        label=kind.value,
    )


def piecewise_ballistic(
    t: FloatArray, t_event: float, anchor: Any, before: tuple[Any, Any], after: tuple[Any, Any]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Position, velocity, acceleration with p(t_event) = anchor and (v, a) switching there."""
    tau = (t - t_event)[:, None]
    late = tau >= 0
    v = np.where(late, np.asarray(after[0]), np.asarray(before[0]))
    a = np.where(late, np.asarray(after[1]), np.asarray(before[1]))
    position = np.asarray(anchor, dtype=np.float64) + v * tau + 0.5 * a * tau**2
    return position, v + a * tau, np.broadcast_to(a, position.shape).copy()


def touching(
    a_id: str,
    a: Solid,
    b_id: str,
    b: Body,
    kind: ContactKind,
    t_start: float,
    t_end: float,
) -> ContactRecord:
    """World-frame contact record of two touching bodies (points from ``contact_manifold``)."""
    manifold = contact_manifold(a, b, CONTACT_TOL)
    if manifold is None:
        raise ValueError(f"{a_id} and {b_id} do not touch")
    normal = manifold.normals.mean(axis=0)
    return ContactRecord(
        object_a=a_id,
        object_b=b_id,
        t_start=float(t_start),
        t_end=float(t_end),
        kind=kind,
        normal=normal / np.linalg.norm(normal),
        points=manifold.points,
        manifold=ManifoldKind.PRIMITIVE,
        point_normals=manifold.normals,
        point_distances=manifold.distances,
        confidence=1.0,
        source="analytic",
    )


def resting(a: SceneObject, b: SceneObject | None = None, frame: int = 0) -> ContactRecord:
    """RESTING contact over the whole scene, posed at native ``frame`` (``b`` None = ground)."""
    other: Body = GROUND if b is None else b.solid_at(b.position[frame])
    other_id = ENV_ID if b is None else b.object_id
    solid = a.solid_at(a.position[frame])
    return touching(a.object_id, solid, other_id, other, ContactKind.RESTING, 0.0, a.frames - 1)


def oracle_scale(
    meters_per_unit: float = 1.0,
    rel_sigma: float = 0.0,
    source: ScaleSource = ScaleSource.GROUND_TRUTH,
) -> EffectiveScale:
    return EffectiveScale(meters_per_unit=meters_per_unit, rel_sigma=rel_sigma, source=source)


def build(
    clip_id: str,
    objects: Sequence[SceneObject],
    records: Iterable[ContactRecord] = (),
    pitch_deg: float = 0.0,
) -> Case:
    """Express a world scene in the camera and wrap it as an oracle ``Case`` in meters."""
    view = camera(pitch_deg)
    states = _camera_states(clip_id, objects, view)
    contacts = ContactTable(
        clip_id=clip_id,
        frame=CoordFrame.CAMERA_OPENCV,
        unit=LengthUnit.METER,
        source_key=states.source_key,
        records=tuple(_record_in(view, r) for r in records),
    )
    clip = _manifest(clip_id, objects[0].frames)
    return Case(clip=clip, states=states, contacts=contacts, scale=oracle_scale(), view=view)


def _manifest(clip_id: str, frames: int) -> ClipManifest:
    return ClipManifest(
        dataset=DATASET,
        clip_id=clip_id,
        split=Split.EVAL,
        video_path=f"{DATASET}/{clip_id}",
        n_frames=frames,
        width=64,
        height=48,
        container_fps=FPS,
        seconds_per_frame=DT,
        time_source=TimeSource.GROUND_TRUTH,
        gt_stages=STAGE_ORDER,
    )


def _camera_states(clip_id: str, objects: Sequence[SceneObject], view: View) -> ObjectStateTable:
    frames, count = objects[0].frames, len(objects)
    frame_index = np.arange(frames, dtype=np.int64)
    blank = ObjectStateTable.blank(
        clip_id=clip_id,
        frame=CoordFrame.CAMERA_OPENCV,
        unit=LengthUnit.METER,
        source_key=ground_truth_source_key(clip_id),
        t=np.asarray(frame_index * DT, dtype=np.float64),
        frame_index=frame_index,
        object_ids=[o.object_id for o in objects],
        labels=[o.label for o in objects],
        primitive=[o.kind.value for o in objects],
    )

    def stacked(values: Iterable[FloatArray]) -> FloatArray:
        return np.stack(list(values), axis=1)

    quaternions = [quaternion_from_matrix(view.rotation @ o.rotation) for o in objects]
    return replace(
        blank,
        position=stacked(view.point(o.position) for o in objects),
        position_sigma=np.zeros((frames, count, 3)),
        velocity=stacked(view.vector(o.velocity) for o in objects),
        velocity_sigma=np.zeros((frames, count, 3)),
        acceleration=stacked(view.vector(o.acceleration) for o in objects),
        acceleration_sigma=np.zeros((frames, count, 3)),
        angular_velocity=np.zeros((frames, count, 3)),
        quaternion=stacked(np.tile(q, (frames, 1)) for q in quaternions),
        orientation_sigma=np.zeros((frames, count)),
        size=stacked(np.broadcast_to(o.extents, (frames, 3)).copy() for o in objects),
        size_sigma=np.zeros((frames, count, 3)),
        occlusion=np.zeros((frames, count)),
        confidence=np.ones((frames, count)),
        gravity_dir=view.vector(DOWN),
        gravity_dir_sigma=0.0,
    )


def _record_in(view: View, record: ContactRecord) -> ContactRecord:
    normals = record.point_normals
    return replace(
        record,
        normal=view.vector(record.normal),
        points=view.point(record.points),
        point_normals=None if normals is None else view.vector(normals),
    )


# Edits.


def strided(case: Case, step: int, quantize: bool) -> Case:
    """Every ``step``-th native frame; with ``quantize`` contact times snap to sampled frames.

    Snapping goes to the nearest sampled frame, ties down for t_start and up for t_end (as the
    extractor emits them); without it the records keep their native bounds (as ``mixed`` inputs do).
    """
    states = take_frames(case.states, case.states.frame_index[::step])
    contacts = case.contacts
    if quantize and contacts is not None:
        sampled = states.frame_index.astype(np.float64)
        records = tuple(
            replace(
                r,
                t_start=_nearest(sampled, r.t_start, up=False),
                t_end=_nearest(sampled, r.t_end, up=True),
            )
            for r in contacts.records
        )
        contacts = replace(contacts, records=records)
    return replace(case, states=states, contacts=contacts)


def _nearest(sampled: FloatArray, value: float, up: bool) -> float:
    gap = np.abs(sampled - value)
    ties = sampled[gap <= gap.min() + 1e-9]
    return float(ties.max() if up else ties.min())


def with_values(case: Case, name: str, object_id: str, frames: Iterable[int], value: Any) -> Case:
    """Set ``states.<name>`` of one object on the given native frames."""
    states = case.states
    array = np.array(getattr(states, name), copy=True)
    rows = np.isin(states.frame_index, np.fromiter(frames, dtype=np.int64))
    array[rows, states.object_index(object_id)] = value
    changes: dict[str, Any] = {name: array}
    return replace(case, states=replace(states, **changes))


def with_sigma(case: Case, **sigmas: float) -> Case:
    """Constant sigma everywhere for the named quantities, e.g. ``position=0.01``."""
    changes: dict[str, Any] = {
        f"{name}_sigma": np.full(getattr(case.states, f"{name}_sigma").shape, value)
        for name, value in sigmas.items()
    }
    return replace(case, states=replace(case.states, **changes))


def with_records(case: Case, *world_records: ContactRecord) -> Case:
    """Append contact records given in world coordinates."""
    assert case.contacts is not None
    added = tuple(_record_in(case.view, r) for r in world_records)
    return replace(case, contacts=replace(case.contacts, records=case.contacts.records + added))


def with_prompt(case: Case, prompt: str) -> Case:
    return replace(case, clip=replace(case.clip, prompt=prompt))


# Scenes.


def overhang(
    offset: float,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    frames: int = FRAMES,
    clip_id: str = "overhang",
) -> Case:
    """Cube "top" (side 0.2) resting on box "base" (0.4 x 0.4 x 0.2) on the ground.

    ``offset`` moves the top along the base's local x; its centre of mass leaves the base
    beyond 0.2 and the two stop touching at 0.3. The whole scene is turned by ``yaw_deg``.
    """
    rotation = yaw(yaw_deg)
    base = still("base", PrimitiveKind.BOX, (0.4, 0.4, 0.2), (0.0, 0.0, 0.1), frames, rotation)
    top_centre = rotation @ np.array([offset, 0.0, 0.0]) + np.array([0.0, 0.0, 0.3])
    top = still("top", PrimitiveKind.BOX, cube(0.2), top_centre, frames, rotation)
    return build(clip_id, [base, top], [resting(base), resting(top, base)], pitch_deg)


def stack(pitch_deg: float = 0.0, frames: int = FRAMES) -> Case:
    return overhang(0.0, pitch_deg=pitch_deg, frames=frames, clip_id="stack")


def floating_box(gap: float = 0.1, pitch_deg: float = 0.0, frames: int = FRAMES) -> Case:
    """Resting cube (side 0.2) ``gap`` meters above the ground, with no contact at all."""
    box = still("box", PrimitiveKind.BOX, cube(0.2), (0.0, 0.0, 0.1 + gap), frames)
    return build("floating_box", [box], [], pitch_deg)


def overlapping_spheres(
    depth: float = 0.05,
    radii: tuple[float, float] = (0.1, 0.15),
    pitch_deg: float = 0.0,
    frames: int = FRAMES,
) -> Case:
    """Two still spheres in the air whose surfaces overlap by ``depth`` (R1 = depth / 0.2)."""
    ra, rb = radii
    distance = ra + rb - depth
    a = still("ball_a", PrimitiveKind.SPHERE, cube(2 * ra), (-distance / 2, 0.0, 1.0), frames)
    b = still("ball_b", PrimitiveKind.SPHERE, cube(2 * rb), (distance / 2, 0.0, 1.0), frames)
    return build("overlapping_spheres", [a, b], [resting(a, b)], pitch_deg)


def box_sphere(depth: float = 0.05, pitch_deg: float = 0.0, frames: int = FRAMES) -> Case:
    """Ball (radius 0.1) sunk ``depth`` into the top face of a cube (side 0.2) on the ground."""
    box = still("box", PrimitiveKind.BOX, cube(0.2), (0.0, 0.0, 0.1), frames)
    centre = (0.0, 0.0, 0.2 + BALL_RADIUS - depth)
    ball = still("ball", PrimitiveKind.SPHERE, cube(2 * BALL_RADIUS), centre, frames)
    return build("box_sphere", [box, ball], [resting(box), resting(ball, box)], pitch_deg)


def sinking_box(depth: float = 0.05, pitch_deg: float = 0.0, frames: int = FRAMES) -> Case:
    """Cube (side 0.2) whose bottom lies ``depth`` below the ground."""
    box = still("box", PrimitiveKind.BOX, cube(0.2), (0.0, 0.0, 0.1 - depth), frames)
    return build("sinking_box", [box], [resting(box)], pitch_deg)


def bounce(
    e: float = 0.8,
    impact_frame: float = EVENT_FRAME,
    kind: PrimitiveKind = PrimitiveKind.SPHERE,
    speed: float = 5.0,
    pitch_deg: float = 0.0,
    frames: int = FRAMES,
) -> Case:
    """Ball (or cube, side 0.2) hitting the ground at ``impact_frame`` with restitution ``e``.

    It arrives at ``speed`` m/s downwards and 0.5 m/s along x under gravity; the defaults keep a
    single impact for e >= 0.7.
    """
    g = STANDARD_GRAVITY * DOWN
    v_in = np.array([0.5, 0.0, -speed])
    v_out = np.array([0.5, 0.0, e * speed])
    anchor = np.array([0.0, 0.0, BALL_RADIUS])
    t_c = impact_frame * DT
    p, v, a = piecewise_ballistic(times(frames), t_c, anchor, (v_in, g), (v_out, g))
    ball = SceneObject("ball", kind, cube(2 * BALL_RADIUS), p, v, a, label=kind.value)
    record = touching(
        "ball",
        ball.solid_at(anchor),
        ENV_ID,
        GROUND,
        ContactKind.IMPACT,
        impact_frame,
        impact_frame,
    )
    return build("bounce", [ball], [record], pitch_deg)


def collision(
    mass_ratio: float = 1.0,
    gain: float = 0.0,
    impact_frame: float = EVENT_FRAME,
    skew_deg: float = 0.0,
    kind: PrimitiveKind = PrimitiveKind.SPHERE,
    pitch_deg: float = 0.0,
    frames: int = FRAMES,
) -> Case:
    """Head-on collision along x of "ball_i" (+1 m/s) and "ball_j" (-1 m/s) without gravity.

    ``mass_ratio`` = m_i / m_j; the outgoing velocities conserve momentum and multiply the
    kinetic energy by 1 + ``gain``. ``skew_deg`` turns ball_j's velocity change about z, so the
    two changes stop being antiparallel.
    """
    r = mass_ratio
    v_i, v_j = 1.0, -1.0
    centre = (r * v_i + v_j) / (r + 1)
    energy = r * v_i**2 + v_j**2
    relative = -math.sqrt(((1 + gain) * energy - (r + 1) * centre**2) * (r + 1) / r)
    out_i, out_j = centre + relative / (r + 1), centre - relative * r / (r + 1)
    dv_j = yaw(skew_deg) @ np.array([out_j - v_j, 0.0, 0.0])
    zero = np.zeros(3)
    t, t_c = times(frames), impact_frame * DT
    before_i, before_j = np.array([v_i, 0.0, 0.0]), np.array([v_j, 0.0, 0.0])
    anchor_i = np.array([-BALL_RADIUS, 0.0, 1.0])
    anchor_j = np.array([BALL_RADIUS, 0.0, 1.0])
    p_i, vel_i, a_i = piecewise_ballistic(
        t, t_c, anchor_i, (before_i, zero), (np.array([out_i, 0.0, 0.0]), zero)
    )
    p_j, vel_j, a_j = piecewise_ballistic(
        t, t_c, anchor_j, (before_j, zero), (before_j + dv_j, zero)
    )
    ball_i = SceneObject("ball_i", kind, cube(2 * BALL_RADIUS), p_i, vel_i, a_i, label=kind.value)
    ball_j = SceneObject("ball_j", kind, cube(2 * BALL_RADIUS), p_j, vel_j, a_j, label=kind.value)
    record = touching(
        "ball_i",
        ball_i.solid_at(anchor_i),
        "ball_j",
        ball_j.solid_at(anchor_j),
        ContactKind.IMPACT,
        impact_frame,
        impact_frame,
    )
    return build("collision", [ball_i, ball_j], [record], pitch_deg)


def rolling_ball(
    speed: float = 0.5, pitch_deg: float = 0.0, frames: int = FRAMES, clip_id: str = "rolling"
) -> Case:
    """Ball (radius 0.1) rolling along x on the ground at constant ``speed``."""
    t = times(frames)
    position = np.column_stack([speed * t, np.zeros(frames), np.full(frames, BALL_RADIUS)])
    velocity = np.tile([speed, 0.0, 0.0], (frames, 1))
    ball = SceneObject(
        "ball",
        PrimitiveKind.SPHERE,
        cube(2 * BALL_RADIUS),
        position,
        velocity,
        np.zeros((frames, 3)),
        label="sphere",
    )
    return build(clip_id, [ball], [resting(ball, frame=frames // 2)], pitch_deg)


def disappearance(
    first: int = 4, last: int = 7, pitch_deg: float = 0.0, frames: int = FRAMES
) -> Case:
    """Rolling ball that is in frame but not visible (and not occluded) on frames first..last."""
    case = rolling_ball(pitch_deg=pitch_deg, frames=frames, clip_id="disappearance")
    return with_values(case, "visible", "ball", range(first, last + 1), False)


def size_change(
    factor: float = 1.3, frame: int = 16, pitch_deg: float = 0.0, frames: int = FRAMES
) -> Case:
    """Rolling ball whose extents are multiplied by ``factor`` from native ``frame`` on."""
    case = rolling_ball(pitch_deg=pitch_deg, frames=frames, clip_id="size_change")
    grown = case.states.size[case.states.frame_index >= frame, 0] * factor
    return with_values(case, "size", "ball", range(frame, frames), grown)


def resized_pair(
    factor: float = 1.3,
    first: int = 12,
    last: int = 17,
    pitch_deg: float = 0.0,
    frames: int = FRAMES,
    balls: int = 2,
) -> Case:
    """``balls`` still balls (radius 0.1) in a row; ball_0's extents are multiplied by ``factor``
    on native frames first..last, a window shorter than half the clip (its median is kept)."""
    objects = [
        still(f"ball_{i}", PrimitiveKind.SPHERE, cube(2 * BALL_RADIUS), (0.4 * i, 0.0, 1.0), frames)
        for i in range(balls)
    ]
    case = build("resized_pair", objects, [], pitch_deg)
    rows = range(first, last + 1)
    grown = case.states.size[first : last + 1, 0] * factor
    return with_values(case, "size", "ball_0", rows, grown)


def teleport(
    delta: float = 0.3,
    frame: int = 16,
    flight: bool = False,
    pitch_deg: float = 0.0,
    frames: int = FRAMES,
) -> Case:
    """Ball displaced by ``delta`` meters along world y from native ``frame`` on.

    ``flight=False`` uses the rolling ball (constant velocity, resting on the ground);
    ``flight=True`` uses the free-flight parabola.
    """
    make = parabola if flight else rolling_ball
    base = make(pitch_deg=pitch_deg, frames=frames, clip_id="teleport")
    shift = base.view.vector(np.array([0.0, delta, 0.0]))
    moved = base.states.position[base.states.frame_index >= frame, 0] + shift
    return with_values(base, "position", "ball", range(frame, frames), moved)


def parabola(
    g: float = STANDARD_GRAVITY,
    g_after: float | None = None,
    change_frame: float = 16,
    pitch_deg: float = 0.0,
    frames: int = FRAMES,
    clip_id: str = "parabola",
) -> Case:
    """Ball in free flight (no contacts) under gravity ``g``, switching to ``g_after`` at
    native ``change_frame`` with continuous position and velocity."""
    t, t_c = times(frames), change_frame * DT
    p0, v0 = np.array([-0.5, 0.0, 1.0]), np.array([1.0, 0.0, 3.0])
    a_before = g * DOWN
    a_after = (g if g_after is None else g_after) * DOWN
    anchor = p0 + v0 * t_c + 0.5 * a_before * t_c**2
    v_c = v0 + a_before * t_c
    p, v, a = piecewise_ballistic(t, t_c, anchor, (v_c, a_before), (v_c, a_after))
    ball = SceneObject("ball", PrimitiveKind.SPHERE, cube(2 * BALL_RADIUS), p, v, a, label="sphere")
    return build(clip_id, [ball], [], pitch_deg)
