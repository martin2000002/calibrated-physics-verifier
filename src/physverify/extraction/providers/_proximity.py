"""Proximity contacts of primitives fitted to the kinematic states: the scene,
its ground plane and the per-pair intervals, shared by the CONTACTS providers (not a plugin)."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from physverify.extraction.providers._common import ProviderParams
from physverify.physics.contact_manifold import contact_manifold
from physverify.physics.geometry import (
    Body,
    Manifold,
    Plane,
    Pose,
    Primitive,
    Solid,
    assumed_upright_pose,
    characteristic_lengths,
    pose_from_quaternion,
)
from physverify.physics.kinematics import visible_runs
from physverify.physics.signed_distance import signed_distance
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ManifoldKind
from physverify.schema.states import ObjectStateTable, PrimitiveKind

FloatArray = NDArray[np.float64]
YAW_AXIS = (1.0, 0.0, 0.0)
"""Assumed yaw reference: the extents were measured along the camera axes."""
SOURCE = "proximity"


class ProximityParams(ProviderParams):
    tol_rel: float = Field(default=0.05, gt=0.0)
    rest_speed_rel: float = Field(default=0.5, ge=0.0)
    min_rest_frames: int = Field(default=3, ge=1)
    normal_split_deg: float = Field(default=20.0, gt=0.0)
    ground: Literal["lowest_rest", "none"] = "lowest_rest"


@dataclass(frozen=True)
class PlacedBody:
    body: Body
    assumed: bool
    point_only: bool


@dataclass(frozen=True)
class FrameContact:
    frame: int
    manifold: Manifold
    normal: FloatArray
    distance: float
    distance_sigma: float
    normal_speed: float


@dataclass(frozen=True)
class Pair:
    a: int
    b: int | None
    tol: float
    length: float


class ContactScene:
    """Per-clip bodies, ground plane and pair tolerances."""

    def __init__(
        self, states: ObjectStateTable, params: ProximityParams, split_on_point_count: bool
    ) -> None:
        self.states = states
        self.params = params
        self.split_on_point_count = split_on_point_count
        self.lengths = object_lengths(states)
        gravity = np.asarray(states.gravity_dir, dtype=np.float64)
        self.gravity = gravity if bool(np.all(np.isfinite(gravity))) else None
        self.bodies = [
            [self._place(t, n) for n in range(states.n_objects)] for t in range(states.n_frames)
        ]
        self.ground = self._ground()

    def records(self) -> Iterator[ContactRecord]:
        for pair in self._pairs():
            contacts = [c for t in range(self.states.n_frames) if (c := self._contact(pair, t))]
            for interval in self._intervals(contacts):
                yield self._record(pair, interval)

    def _pairs(self) -> Iterator[Pair]:
        order = np.argsort(self.states.object_ids)
        tol_rel = self.params.tol_rel
        for a, b in combinations(order.tolist(), 2):
            length = float(min(self.lengths[a], self.lengths[b]))
            yield Pair(a=a, b=b, tol=tol_rel * length, length=length)
        if self.ground is not None:
            for a in order.tolist():
                length = float(self.lengths[a])
                yield Pair(a=a, b=None, tol=tol_rel * length, length=length)

    def _place(self, t: int, n: int) -> PlacedBody | None:
        states = self.states
        position, size = states.position[t, n], states.size[t, n]
        known = np.all(np.isfinite(position)) and np.all(np.isfinite(size)) and np.all(size > 0)
        if not (states.visible[t, n] and known):
            return None
        quaternion = states.quaternion[t, n]
        kind = PrimitiveKind(str(states.primitive[n]))
        if bool(np.all(np.isfinite(quaternion))):
            pose = pose_from_quaternion(position, quaternion)
            return PlacedBody(Solid(Primitive(kind, size), pose), assumed=False, point_only=False)
        if self.gravity is not None:
            pose = assumed_upright_pose(position, self.gravity, YAW_AXIS)
            # Extents were measured along camera axes; re-express them along the assumed axes.
            local = np.sqrt((pose.rotation**2).T @ size**2)
            return PlacedBody(Solid(Primitive(kind, local), pose), assumed=True, point_only=False)
        diameter = float(characteristic_lengths(size))
        sphere = Primitive(PrimitiveKind.SPHERE, np.full(3, diameter))
        return PlacedBody(Solid(sphere, Pose(position, np.eye(3))), assumed=True, point_only=True)

    def _ground(self) -> Plane | None:
        """Plane under the lowest object at rest: h = min over rest frames of p.up - h_up."""
        if self.params.ground == "none" or self.gravity is None:
            return None
        up = -self.gravity / float(np.linalg.norm(self.gravity))
        reference = Plane.ground(0.0, up)
        heights = [
            signed_distance(placed.body, reference).distance
            for t, n in self._rest_cells()
            if (placed := self.bodies[t][n]) is not None
        ]
        return Plane.ground(min(heights), up) if heights else None

    def _rest_cells(self) -> Iterator[tuple[int, int]]:
        speed = np.linalg.norm(self.states.velocity, axis=2)
        for n in range(self.states.n_objects):
            limit = self.params.rest_speed_rel * float(self.lengths[n])
            with np.errstate(invalid="ignore"):
                rest = speed[:, n] <= limit
            for run in visible_runs(rest):
                if run.stop - run.start >= self.params.min_rest_frames:
                    yield from ((t, n) for t in range(run.start, run.stop))

    def _contact(self, pair: Pair, t: int) -> FrameContact | None:
        placed_a = self.bodies[t][pair.a]
        placed_b = None if pair.b is None else self.bodies[t][pair.b]
        if placed_a is None or (pair.b is not None and placed_b is None):
            return None
        other: Body | None = self.ground if placed_b is None else placed_b.body
        if other is None:
            return None
        manifold = contact_manifold(placed_a.body, other, pair.tol)
        if manifold is None:
            return None
        normal = unit_mean(manifold.normals)
        velocity = self.states.velocity[t, pair.a] - self._velocity(t, pair.b)
        variance = self.states.position_sigma[t, pair.a] ** 2 + self._sigma(t, pair.b) ** 2
        return FrameContact(
            frame=t,
            manifold=manifold,
            normal=normal,
            distance=float(manifold.distances.min()),
            distance_sigma=math.sqrt(float(normal**2 @ variance)),
            normal_speed=abs(float(velocity @ normal)),
        )

    def _velocity(self, t: int, n: int | None) -> FloatArray:
        return np.zeros(3) if n is None else self.states.velocity[t, n]

    def _sigma(self, t: int, n: int | None) -> FloatArray:
        return np.zeros(3) if n is None else self.states.position_sigma[t, n]

    def _intervals(self, contacts: Sequence[FrameContact]) -> Iterator[list[FrameContact]]:
        """Consecutive frames, split when the normal turns (and, with ``split_on_point_count``,
        when the point count changes)."""
        limit = math.cos(math.radians(self.params.normal_split_deg))
        current: list[FrameContact] = []
        for contact in contacts:
            if current and not continues(current[-1], contact, limit, self.split_on_point_count):
                yield current
                current = []
            current.append(contact)
        if current:
            yield current

    def _record(self, pair: Pair, interval: list[FrameContact]) -> ContactRecord:
        middle = interval[len(interval) // 2]
        frames = self.states.frame_index
        placed = [self.bodies[middle.frame][pair.a]]
        if pair.b is not None:
            placed.append(self.bodies[middle.frame][pair.b])
        speed = nan_median([c.normal_speed for c in interval])
        resting = (
            len(interval) >= self.params.min_rest_frames
            and speed <= self.params.rest_speed_rel * pair.length
        )
        point_only = any(p is not None and p.point_only for p in placed)
        return ContactRecord(
            object_a=str(self.states.object_ids[pair.a]),
            object_b=ENV_ID if pair.b is None else str(self.states.object_ids[pair.b]),
            t_start=float(frames[interval[0].frame]),
            t_end=float(frames[interval[-1].frame]),
            kind=ContactKind.RESTING if resting else ContactKind.IMPACT,
            normal=middle.normal,
            points=middle.manifold.points,
            manifold=ManifoldKind.POINT if point_only else ManifoldKind.PRIMITIVE,
            point_normals=middle.manifold.normals,
            point_distances=middle.manifold.distances,
            orientation_assumed=any(p is not None and p.assumed for p in placed),
            max_penetration=max(0.0, -min(c.distance for c in interval)),
            confidence=interval_confidence(interval, pair.tol),
            source=SOURCE,
        )


def object_lengths(states: ObjectStateTable) -> FloatArray:
    """Median characteristic length L of each object over the frames where it is known."""
    lengths = characteristic_lengths(states.size)
    lengths[~states.visible] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median: FloatArray = np.nanmedian(lengths, axis=0)
    return median


def nan_median(values: Sequence[float]) -> float:
    """``np.nanmedian`` without the all-NaN warning: NaN when every value is NaN."""
    array = np.asarray(values, dtype=np.float64)
    known = array[~np.isnan(array)]
    return float(np.median(known)) if known.size else math.nan


def unit_mean(normals: FloatArray) -> FloatArray:
    mean = normals.mean(axis=0)
    unit: FloatArray = mean / float(np.linalg.norm(mean))
    return unit


def continues(
    previous: FrameContact, contact: FrameContact, cos_limit: float, split_on_point_count: bool
) -> bool:
    same_count = contact.manifold.points.shape[0] == previous.manifold.points.shape[0]
    return (
        contact.frame == previous.frame + 1
        and (same_count or not split_on_point_count)
        and float(contact.normal @ previous.normal) >= cos_limit
    )


def contact_probability(distance: float, tol: float, sigma: float) -> float:
    """Phi((tol - d) / sigma_d); the sigma -> 0 limit is the step 1[d <= tol]."""
    if math.isnan(sigma):
        return math.nan
    if sigma == 0.0:
        return 1.0 if distance <= tol else 0.0
    return 0.5 * (1.0 + math.erf((tol - distance) / (sigma * math.sqrt(2.0))))


def interval_confidence(interval: Sequence[FrameContact], tol: float) -> float:
    values = [contact_probability(c.distance, tol, c.distance_sigma) for c in interval]
    return math.nan if any(math.isnan(v) for v in values) else float(np.median(values))
