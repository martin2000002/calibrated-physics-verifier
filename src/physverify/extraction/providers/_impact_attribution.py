"""Attribution of kinematic impacts and their contact records (ADR 0005 4.9, it3); not a plugin.

Each impact of an object goes to the candidate surface whose normal best aligns with its
velocity change, with a positive cosine (the impulse pushes the object off the surface):
(a) another object with an opposite impact within one row step whose image boxes touch, normal
along the difference of the two velocity changes (momentum conservation; the image test needs no
depth, which is what is least known between objects); (b) another object with a proximity record
around the impact, its normal; (c) the environment when gravity is known, normal up. Walls and
ceilings are not modelled. A pair accepted through (a) consumes the impacts of both objects within
one row step, so neither adds an environment record there (J7).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.extraction.providers._impacts import Bases, Impact, track_impacts
from physverify.extraction.providers._proximity import ContactScene, nan_median
from physverify.physics.geometry import Plane, Pose, Solid, characteristic_lengths
from physverify.physics.signed_distance import signed_distance
from physverify.physics.tracks import (
    MIN_NOISE_STENCILS,
    NoiseScale,
    contact_window,
    noise_scale,
    pixel_floor,
    row_step,
    track_noise,
)
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable, ManifoldKind
from physverify.schema.perception import CameraModel
from physverify.schema.states import CoordFrame, ObjectStateTable

FloatArray = NDArray[np.float64]
SOURCE = "kinematic"
PAIR_MARGIN = 1
"""A proximity record is around an impact when its contact window widened by one native frame
(``contact_rows(margin=1)``) holds the impact's frame."""

EventKey = tuple[int, int]


@dataclass(frozen=True)
class Event:
    """Impact of object ``index``."""

    index: int
    impact: Impact

    @property
    def key(self) -> EventKey:
        return (self.index, self.impact.row)


@dataclass(frozen=True)
class Choice:
    """The surface an event is attributed to: ``other`` (None = environment) with ``normal``
    from that surface to the event's object; ``partner`` is the other object's opposite impact
    of a candidate (a)."""

    event: Event
    other: int | None
    normal: FloatArray
    cosine: float
    partner: Event | None = None


@dataclass(frozen=True)
class KinematicScene:
    """A clip's states with what the attribution reads: the proximity scene (bodies, gravity),
    its records, the positions in the camera frame (None when they cannot be had) and time."""

    states: ObjectStateTable
    proximity: ContactScene
    records: tuple[ContactRecord, ...]
    camera_positions: FloatArray | None
    seconds_per_frame: float
    tol_rel: float

    @property
    def step(self) -> int:
        return row_step(self.states.frame_index)

    @property
    def row_seconds(self) -> float:
        return self.step * self.seconds_per_frame


def camera_positions(states: ObjectStateTable, camera: CameraModel) -> FloatArray | None:
    """Positions in CAMERA_OPENCV: as they are, or moved with the camera pose (one per row or
    one for the clip); None without a pose that fits the rows."""
    if states.frame is CoordFrame.CAMERA_OPENCV:
        return np.asarray(states.position, dtype=np.float64)
    pose = camera.world_from_camera
    if pose is None:
        return None
    poses = pose if pose.ndim == 3 else np.broadcast_to(pose, (states.n_frames, 4, 4))
    if poses.shape[0] != states.n_frames:
        return None
    camera_from_world = np.linalg.inv(poses)
    rotation, translation = camera_from_world[:, :3, :3], camera_from_world[:, :3, 3]
    moved = np.einsum("tij,tnj->tni", rotation, states.position) + translation[:, None, :]
    return np.asarray(moved, dtype=np.float64)


def object_sigma(
    states: ObjectStateTable, index: int, depths: FloatArray | None, focal: float
) -> NoiseScale | None:
    """sigma-hat per axis of one object: its track noise floored by one pixel at its median
    camera depth; None without either (no impacts are then sought for it)."""
    depth = nan_median(depths.tolist()) if depths is not None else math.nan
    empty = ContactTable(
        clip_id=states.clip_id,
        frame=states.frame,
        unit=states.unit,
        source_key=states.source_key,
        records=(),
    )
    noise = track_noise(states, empty, index, MIN_NOISE_STENCILS)
    return noise_scale(noise, pixel_floor(depth, focal))


def detect(scene: KinematicScene, focal: float, side_rows: int, alpha: float) -> list[Event]:
    states, cam = scene.states, scene.camera_positions
    bases: Bases = {}
    events: list[Event] = []
    for n in range(states.n_objects):
        noise = object_sigma(states, n, None if cam is None else cam[:, n, 2], focal)
        if noise is None:
            continue
        found = track_impacts(states.t, states.position[:, n], noise, side_rows, alpha, bases)
        events.extend(Event(n, impact) for impact in found)
    return events


def nearest_row(scene: KinematicScene, time: float) -> int:
    return int(np.argmin(np.abs(scene.states.t - time)))


def cosine(vector: FloatArray, normal: FloatArray) -> float:
    norm = float(np.linalg.norm(vector))
    return float(vector @ normal) / norm if norm > 0.0 else math.nan


def unit(vector: FloatArray) -> FloatArray | None:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else None


def image_box(scene: KinematicScene, index: int, row: int) -> FloatArray | None:
    """(x/z, y/z, half-size) of the object's bounding sphere dilated by tol_rel L, in image
    coordinates; a projected box never leaves it, so the test misses no real contact."""
    if scene.camera_positions is None:
        return None
    x, y, z = scene.camera_positions[row, index]
    size = scene.states.size[row, index]
    length = float(characteristic_lengths(size))
    if not (math.isfinite(z) and z > 0.0 and math.isfinite(length)):
        return None
    half = (0.5 * float(np.linalg.norm(size)) + scene.tol_rel * length) / z
    return np.array([x / z, y / z, half])


def boxes_touch(scene: KinematicScene, a: int, b: int, time: float) -> bool:
    row = nearest_row(scene, time)
    box_a, box_b = image_box(scene, a, row), image_box(scene, b, row)
    if box_a is None or box_b is None:
        return False
    reach = box_a[2] + box_b[2]
    return bool(np.all(np.abs(box_a[:2] - box_b[:2]) <= reach))


def impulse_candidates(
    scene: KinematicScene, event: Event, events: Sequence[Event]
) -> Iterator[Choice]:
    """(a): opposite impacts of other objects within one row step whose image boxes touch."""
    dv = event.impact.delta_v
    for other in events:
        if other.index == event.index:
            continue
        close = abs(other.impact.time - event.impact.time) <= scene.row_seconds
        if not close or not float(dv @ other.impact.delta_v) < 0.0:
            continue
        normal = unit(dv - other.impact.delta_v)
        if normal is not None and boxes_touch(scene, event.index, other.index, event.impact.time):
            yield Choice(event, other.index, normal, cosine(dv, normal), other)


def proximity_candidates(scene: KinematicScene, event: Event) -> Iterator[Choice]:
    """(b): proximity records of a pair with the event's object around the impact."""
    ids = scene.states.object_ids
    own = str(ids[event.index])
    frame = event.impact.time / scene.seconds_per_frame
    for record in scene.records:
        if own not in (record.object_a, record.object_b) or ENV_ID in (
            record.object_a,
            record.object_b,
        ):
            continue
        first, last = contact_window(record, scene.step, PAIR_MARGIN)
        if first <= frame <= last:
            other = record.object_b if record.object_a == own else record.object_a
            normal = record.normal if record.object_a == own else -record.normal
            index = int(np.flatnonzero(ids == other)[0])
            yield Choice(event, index, normal, cosine(event.impact.delta_v, normal))


def environment_candidate(scene: KinematicScene, event: Event) -> Iterator[Choice]:
    """(c): the environment, normal up, when gravity is known."""
    gravity = scene.proximity.gravity
    if gravity is not None:
        up = -gravity / float(np.linalg.norm(gravity))
        yield Choice(event, None, up, cosine(event.impact.delta_v, up))


def choose(scene: KinematicScene, event: Event, events: Sequence[Event]) -> Choice | None:
    """The candidate with the largest cosine, if that cosine is positive."""
    candidates = [
        *impulse_candidates(scene, event, events),
        *proximity_candidates(scene, event),
        *environment_candidate(scene, event),
    ]
    usable = [c for c in candidates if math.isfinite(c.cosine) and c.cosine > 0.0]
    return max(usable, key=lambda c: c.cosine) if usable else None


def resolve(scene: KinematicScene, events: Sequence[Event]) -> list[Choice]:
    """Final choices. The pair through (a) with the largest cosine is accepted and consumes both
    objects' impacts within one row step; the events left choose again among themselves, so an
    event whose best partner was consumed falls back to its next surface (three bodies meeting
    at once, audit L4). Without pairs left, the remaining events keep their choices."""
    consumed: set[EventKey] = set()
    final: list[Choice] = []
    while True:
        left = [e for e in events if e.key not in consumed]
        choices = [c for e in left if (c := choose(scene, e, left)) is not None]
        pairs = [c for c in choices if c.partner is not None]
        if not pairs:
            return final + choices
        best = max(pairs, key=lambda c: c.cosine)
        final.append(best)
        consumed |= around(scene, events, best)


def around(scene: KinematicScene, events: Sequence[Event], pair: Choice) -> set[EventKey]:
    """Events of the pair's two objects within one row step of its impact."""
    objects = {pair.event.index, pair.other}
    time = pair.event.impact.time
    return {
        e.key
        for e in events
        if e.index in objects and abs(e.impact.time - time) <= scene.row_seconds
    }


def facing_point(
    scene: KinematicScene, index: int, row: int, position: FloatArray, normal: FloatArray
) -> FloatArray | None:
    """Point of the object's primitive (pose of ``row``, centre moved to ``position``) with the
    smallest normal . x; None when the object cannot be placed at that row."""
    placed = scene.proximity.bodies[row][index]
    if placed is None or not isinstance(placed.body, Solid):
        return None
    solid = Solid(placed.body.primitive, Pose(np.asarray(position), placed.body.pose.rotation))
    return signed_distance(solid, Plane(normal=normal, offset=0.0)).point_a


def assumed_orientation(scene: KinematicScene, row: int, indices: Sequence[int]) -> bool:
    bodies = [scene.proximity.bodies[row][n] for n in indices]
    return any(b is not None and b.assumed for b in bodies)


def record_of(scene: KinematicScene, choice: Choice) -> ContactRecord | None:
    """The IMPACT record of a choice; a pair is ordered as the proximity provider orders it
    (object ids sorted), with the normal from object_b to object_a."""
    impact, index = choice.event.impact, choice.event.index
    row = nearest_row(scene, impact.time)
    own = facing_point(scene, index, row, impact.position, choice.normal)
    if own is None:
        return None
    if choice.other is None:
        return make_record(scene, (index, None), impact.time, choice.normal, own, row)
    partner = choice.partner
    other_at = (
        scene.states.position[row, choice.other] if partner is None else partner.impact.position
    )
    facing = facing_point(scene, choice.other, row, other_at, -choice.normal)
    if facing is None:
        return None
    time = impact.time if partner is None else 0.5 * (impact.time + partner.impact.time)
    ids = scene.states.object_ids
    first = str(ids[index]) < str(ids[choice.other])
    pair = (index, choice.other) if first else (choice.other, index)
    normal = choice.normal if first else -choice.normal
    return make_record(scene, pair, time, normal, 0.5 * (own + facing), row)


def make_record(
    scene: KinematicScene,
    pair: tuple[int, int | None],
    time: float,
    normal: FloatArray,
    point: FloatArray,
    row: int,
) -> ContactRecord:
    """One-point IMPACT at the fractional native frame of ``time``; no probability is known,
    so the confidence is NaN, and the point lies on the surfaces (distance 0, J6)."""
    a, b = pair
    ids = scene.states.object_ids
    frame = time / scene.seconds_per_frame
    return ContactRecord(
        object_a=str(ids[a]),
        object_b=ENV_ID if b is None else str(ids[b]),
        t_start=frame,
        t_end=frame,
        kind=ContactKind.IMPACT,
        normal=np.asarray(normal, dtype=np.float64),
        points=np.asarray(point, dtype=np.float64)[None, :],
        manifold=ManifoldKind.POINT,
        point_normals=np.asarray(normal, dtype=np.float64)[None, :],
        point_distances=np.zeros(1),
        orientation_assumed=assumed_orientation(scene, row, [n for n in pair if n is not None]),
        confidence=math.nan,
        source=SOURCE,
    )


def merge(
    proximity: Sequence[ContactRecord], kinematic: Sequence[ContactRecord], step: int
) -> tuple[ContactRecord, ...]:
    """Every proximity record but the IMPACTs of a pair whose window (margin 1) holds the frame
    of a kinematic impact of that pair, which replaces them; then the kinematic records."""

    def replaced(record: ContactRecord) -> bool:
        if record.kind is not ContactKind.IMPACT:
            return False
        first, last = contact_window(record, step, PAIR_MARGIN)
        pair = {record.object_a, record.object_b}
        return any(
            {k.object_a, k.object_b} == pair and first <= k.t_start <= last for k in kinematic
        )

    return (*(r for r in proximity if not replaced(r)), *kinematic)


def kinematic_records(
    scene: KinematicScene, focal: float, side_rows: int, alpha: float
) -> tuple[ContactRecord, ...]:
    """Proximity records merged with the attributed kinematic impacts."""
    choices = resolve(scene, detect(scene, focal, side_rows, alpha))
    records = [r for c in choices if (r := record_of(scene, c)) is not None]
    records.sort(key=lambda r: (r.object_a, r.object_b, r.t_start))
    return merge(scene.records, records, scene.step)
