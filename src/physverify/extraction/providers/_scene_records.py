"""Contact records of the kinematic scene provider; not a plugin.

Existence comes from a rest test (``_rest_existence.py``), a pair rest test (``_pair_rest.py``)
and an impact test (``_impact_existence.py``), each injected. RESTING on the environment: rest
rows of an object not resting on another one, with the optional gap gate: within K sigma_gap of
its scene plane, K = z(alpha / 2) and sigma_gap the profile's measured gap error. The scene plane
of each object rests on the others' rest runs and significant upward 3D velocity kinks.
Environment records place their points on the body's own bottom with distance 0, so that
``support_plane`` returns where the body is, not the scene plane it was tested against.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from physverify.extraction.providers._impact_attribution import (
    Choice,
    Event,
    KinematicScene,
    detect,
    facing_point,
    make_record,
    nearest_row,
)
from physverify.extraction.providers._proximity import FrameContact, Pair
from physverify.extraction.providers._scene_plane import (
    GapError,
    HeightError,
    ScenePlane,
    SupportPlane,
    height,
    median_positions,
    moved,
    solid,
    support_gap,
    up_of,
)
from physverify.extraction.providers._velocity_tests import (
    NormalChange,
    long_runs,
    two_sided_quantile,
)
from physverify.physics.contact_manifold import contact_manifold
from physverify.physics.geometry import Plane, Solid
from physverify.physics.signed_distance import signed_distance
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ManifoldKind

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
SOURCE = "kinematic_scene"
Interval = tuple[Pair, list[FrameContact]]


@dataclass(frozen=True)
class SceneInputs:
    """A clip's kinematic scene (pair proximity only), its pair intervals, its rest rows (T, N)
    and whether their test was known, sigma_v per object, the lift's height and gap errors and
    the test settings."""

    scene: KinematicScene
    intervals: tuple[Interval, ...]
    rest: BoolArray
    rest_known: bool
    sigma_v: FloatArray
    height_error: HeightError
    gap_error: GapError
    alpha: float
    focal: float
    side_s: float
    env_gap_gate: bool
    impacts: ImpactExistence
    pair_rest: PairRest


@dataclass(frozen=True)
class SceneEvidence:
    """What the impact test may read: up (None without gravity), each object's rest runs on the
    environment, the 3D velocity kinks with their count per object, and the scene planes."""

    up: FloatArray | None
    env_runs: tuple[tuple[slice, ...], ...]
    events: tuple[Event, ...]
    tests: Counter[int]
    planes: tuple[SupportPlane | None, ...]


class ImpactExistence(Protocol):
    def impacts(self, inputs: SceneInputs, evidence: SceneEvidence) -> FoundImpacts: ...


class PairRest(Protocol):
    def pairs(
        self, inputs: SceneInputs, up: FloatArray | None, covered: BoolArray
    ) -> list[ContactRecord]:
        """RESTING pair records; marks in ``covered`` the rows where a body rests on another."""
        ...


@dataclass(frozen=True)
class TestedChoice:
    """An attributed impact, its change along the normal (None: no scene plane) and verdict."""

    choice: Choice
    change: NormalChange | None
    accepted: bool


@dataclass(frozen=True)
class FoundImpacts:
    """IMPACT records and the 3D choices tested on the way (none for a test without them)."""

    records: tuple[ContactRecord, ...]
    tested: tuple[TestedChoice, ...] = ()


@dataclass(frozen=True)
class SceneOutcome:
    """Records and the tests behind them. ``rest_known`` False means no rest could be tested
    (unknown error of the rest test); ``gap_known`` False means the gap error is unknown, so with
    the gap gate on no environment rest could be tested. Neither means that none was found."""

    records: tuple[ContactRecord, ...]
    tested: tuple[TestedChoice, ...]
    planes: tuple[SupportPlane | None, ...]
    rest: BoolArray
    rest_known: bool
    gap_known: bool


def scene_contacts(inputs: SceneInputs) -> SceneOutcome:
    scene, alpha = inputs.scene, inputs.alpha
    states, proximity = scene.states, scene.proximity
    up = up_of(proximity.gravity)
    rest = inputs.rest
    covered = np.zeros(rest.shape, dtype=np.bool_)
    pairs = inputs.pair_rest.pairs(inputs, up, covered)
    events = tuple(detect(scene, inputs.focal, inputs.side_s, alpha, bonferroni=False))
    tests = Counter(e.index for e in events)
    env_runs = tuple(
        long_runs(rest[:, n] & ~covered[:, n], proximity.min_rest_rows)
        for n in range(states.n_objects)
    )
    planes = scene_planes(inputs, up, env_runs, events, tests)
    found = inputs.impacts.impacts(inputs, SceneEvidence(up, env_runs, events, tests, planes))
    alone = rest & ~covered
    near = near_plane(inputs, planes, alone) if inputs.env_gap_gate else alone
    resting = [
        r
        for n, support in enumerate(planes)
        if support is not None
        for run in long_runs(near[:, n], proximity.min_rest_rows)
        if (r := env_resting(inputs, n, run, support))
    ]
    records = sorted(
        [*pairs, *found.records, *resting], key=lambda r: (r.object_a, r.object_b, r.t_start)
    )
    return SceneOutcome(
        tuple(records), found.tested, planes, rest, inputs.rest_known, inputs.gap_error.known
    )


def near_plane(
    inputs: SceneInputs, planes: Sequence[SupportPlane | None], candidates: BoolArray
) -> BoolArray:
    """Candidate rows whose body lies within K sigma_gap of its plane, K = z(alpha / 2);
    none when sigma_gap is unknown. The planes come from the ungated runs, so the gate never
    feeds the plane it tests against."""
    near = np.zeros(candidates.shape, dtype=np.bool_)
    if not inputs.gap_error.known:
        return near
    proximity = inputs.scene.proximity
    k = two_sided_quantile(inputs.alpha)
    for t, n in zip(*np.nonzero(candidates), strict=True):
        support = planes[n]
        if support is None:
            continue
        bound = k * inputs.gap_error.sigma(float(proximity.lengths[n]))
        near[t, n] = abs(support_gap(proximity.bodies[t][n], support)) <= bound
    return near


def scene_planes(
    inputs: SceneInputs,
    up: FloatArray | None,
    env_runs: Sequence[tuple[slice, ...]],
    events: Sequence[Event],
    tests: Counter[int],
) -> tuple[SupportPlane | None, ...]:
    """P_n of every object: evidence from rest runs on the environment and upward impacts."""
    if up is None:
        return (None,) * inputs.scene.states.n_objects
    evidence = rest_heights(inputs, up, env_runs)
    bodies = inputs.scene.proximity.bodies
    for event in events:
        n, impact = event.index, event.impact
        change = NormalChange.along(impact.delta_v, up, float(inputs.sigma_v[n]), tests[n])
        if change.significant(inputs.alpha):
            body = bodies[nearest_row(inputs.scene, impact.time)][n]
            evidence[n].append(height(body, up, impact.position))
    return planes_from(inputs, up, evidence)


def rest_heights(
    inputs: SceneInputs, up: FloatArray, env_runs: Sequence[tuple[slice, ...]]
) -> dict[int, list[float]]:
    """Lowest heights of every object over its rest runs on the environment."""
    bodies = inputs.scene.proximity.bodies
    evidence: dict[int, list[float]] = {n: [] for n in range(inputs.scene.states.n_objects)}
    for n, runs in enumerate(env_runs):
        for run in runs:
            evidence[n].extend(height(bodies[t][n], up) for t in range(run.start, run.stop))
    return evidence


def planes_from(
    inputs: SceneInputs, up: FloatArray, evidence: Mapping[int, Sequence[float]]
) -> tuple[SupportPlane | None, ...]:
    """P_n of every object from the evidence heights of the others."""
    states = inputs.scene.states
    lengths = inputs.scene.proximity.lengths
    sigmas = np.array([inputs.height_error.sigma(float(length)) for length in lengths])
    plane = ScenePlane.of(
        up,
        {n: [h for h in heights if math.isfinite(h)] for n, heights in evidence.items()},
        median_positions(states.position, states.visible),
        sigmas,
        float(states.gravity_dir_sigma),
    )
    return tuple(plane.for_object(n) for n in range(states.n_objects))


def env_impact(
    scene: KinematicScene, event: Event, support: SupportPlane, change: NormalChange
) -> ContactRecord | None:
    """One point at b, the body's lowest point at the impact, with distance 0."""
    up, impact = support.plane.normal, event.impact
    row = nearest_row(scene, impact.time)
    bottom = facing_point(scene, event.index, row, impact.position, up)
    if bottom is None:
        return None
    record = make_record(scene, (event.index, None), impact.time, up, bottom, row)
    return replace(
        record,
        point_distances=np.zeros(1),
        confidence=change.confidence,
        source=SOURCE,
    )


def env_resting(
    inputs: SceneInputs, n: int, run: slice, support: SupportPlane
) -> ContactRecord | None:
    """RESTING on P_n over ``run``, placed on the body's own floor (the median of its lowest
    heights): the patch of the central row's body moved onto that floor, distances 0. The gap
    to P_n only sets the confidence, the two-sided p-value of its median under sigma_gap."""
    scene = inputs.scene
    states, proximity = scene.states, scene.proximity
    up = support.plane.normal
    rows = [t for t in range(run.start, run.stop) if solid(proximity.bodies[t][n]) is not None]
    if not rows:
        return None
    lowest = [height(proximity.bodies[t][n], up) for t in rows]
    gaps = [support_gap(proximity.bodies[t][n], support) for t in rows]
    floor = Plane.ground(float(np.median(lowest)), up)
    placed = proximity.bodies[rows[len(rows) // 2]][n]
    body = solid(placed)
    assert placed is not None and body is not None
    length = float(proximity.lengths[n])
    patch = floor_patch(body, floor, scene.tol_rel * length)
    count = patch.shape[0]
    spread = inputs.gap_error.sigma(length)
    frames = states.frame_index
    return ContactRecord(
        object_a=str(states.object_ids[n]),
        object_b=ENV_ID,
        t_start=float(frames[rows[0]]),
        t_end=float(frames[rows[-1]]),
        kind=ContactKind.RESTING,
        normal=up,
        points=patch,
        manifold=ManifoldKind.POINT if placed.point_only else ManifoldKind.PRIMITIVE,
        point_normals=np.tile(up, (count, 1)),
        point_distances=np.zeros(count),
        orientation_assumed=placed.assumed,
        max_penetration=0.0,
        confidence=two_sided(float(np.median(gaps)), spread),
        source=SOURCE,
    )


def two_sided(value: float, sigma: float) -> float:
    """2 P(Z > |value| / sigma); NaN when both are 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.float64(abs(value)) / np.float64(sigma)
    return float(2.0 * norm.sf(score))


def floor_patch(body: Solid, floor: Plane, tol: float) -> FloatArray:
    """Contact patch of the body moved along up onto ``floor``, projected on it."""
    up = floor.normal
    touching = moved(body, body.pose.position - signed_distance(body, floor).distance * up)
    manifold = contact_manifold(touching, floor, tol)
    points = (
        signed_distance(touching, floor).point_a[None, :] if manifold is None else manifold.points
    )
    on_floor: FloatArray = points - (points @ up - floor.offset)[:, None] * up
    return on_floor
