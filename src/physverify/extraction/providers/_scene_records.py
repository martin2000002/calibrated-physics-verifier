"""Contact records of the kinematic scene provider; not a plugin.

Existence is kinematic. RESTING: rest runs of both objects of a proximity pair (pair record), or
rest rows of an object not resting on another one and within K sigma_gap of its scene plane
(environment record), K = z(alpha / 2) and sigma_gap the profile's measured gap error.
IMPACT: a velocity kink whose shape explains its window best, attributed to a surface as in
``_impact_attribution.py`` and kept when its change along the surface normal is significant.
Environment records place their points on the body's own bottom with distance 0, so that
``support_plane`` returns where the body is, not the scene plane it was tested against.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace

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
    record_of,
    resolve,
)
from physverify.extraction.providers._proximity import FrameContact, Pair, nan_median
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
from physverify.extraction.providers._velocity_tests import NormalChange, long_runs, rest_mask
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
    """A clip's kinematic scene (pair proximity only), its pair intervals, sigma_v per object,
    the lift's height and gap errors and the test settings."""

    scene: KinematicScene
    intervals: tuple[Interval, ...]
    sigma_v: FloatArray
    height_error: HeightError
    gap_error: GapError
    alpha: float
    focal: float
    side_s: float


@dataclass(frozen=True)
class TestedChoice:
    """An attributed impact, its change along the normal (None: no scene plane) and verdict."""

    choice: Choice
    change: NormalChange | None
    accepted: bool


@dataclass(frozen=True)
class SceneOutcome:
    """Records and the tests behind them; ``gap_known`` False means no environment rest could
    be tested (unknown gap error), not that none was found."""

    records: tuple[ContactRecord, ...]
    tested: tuple[TestedChoice, ...]
    planes: tuple[SupportPlane | None, ...]
    rest: BoolArray
    gap_known: bool


def scene_contacts(inputs: SceneInputs) -> SceneOutcome:
    scene, alpha = inputs.scene, inputs.alpha
    states, proximity = scene.states, scene.proximity
    up = up_of(proximity.gravity)
    rest = rest_mask(states, inputs.sigma_v, alpha, proximity.min_rest_rows)
    covered = np.zeros(rest.shape, dtype=np.bool_)
    pairs = list(resting_pairs(scene, inputs.intervals, rest, up, covered))
    events = detect(scene, inputs.focal, inputs.side_s, alpha, bonferroni=False)
    tests = Counter(e.index for e in events)
    env_runs = [
        long_runs(rest[:, n] & ~covered[:, n], proximity.min_rest_rows)
        for n in range(states.n_objects)
    ]
    planes = scene_planes(inputs, up, env_runs, events, tests)
    tested = [gate(inputs, c, tests, planes) for c in resolve(scene, events)]
    impacts = [r for t in tested if t.accepted and (r := impact_record(scene, t, planes))]
    near = near_plane(inputs, planes, rest & ~covered)
    resting = [
        r
        for n, support in enumerate(planes)
        if support is not None
        for run in long_runs(near[:, n], proximity.min_rest_rows)
        if (r := env_resting(inputs, n, run, support))
    ]
    records = sorted(
        [*pairs, *impacts, *resting], key=lambda r: (r.object_a, r.object_b, r.t_start)
    )
    return SceneOutcome(tuple(records), tuple(tested), planes, rest, inputs.gap_error.known)


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
    k = float(norm.isf(inputs.alpha / 2.0))
    for t, n in zip(*np.nonzero(candidates), strict=True):
        support = planes[n]
        if support is None:
            continue
        bound = k * inputs.gap_error.sigma(float(proximity.lengths[n]))
        near[t, n] = abs(support_gap(proximity.bodies[t][n], support)) <= bound
    return near


def resting_pairs(
    scene: KinematicScene,
    intervals: Sequence[Interval],
    rest: BoolArray,
    up: FloatArray | None,
    covered: BoolArray,
) -> Iterator[ContactRecord]:
    """RESTING pair records over the stretches where both objects rest; marks in ``covered``
    the rows where the upper object (larger up . median position) rests on the pair."""
    min_rows = scene.proximity.min_rest_rows
    for pair, interval in intervals:
        assert pair.b is not None
        both = [c for c in interval if rest[c.frame, pair.a] and rest[c.frame, pair.b]]
        for stretch in consecutive(both):
            if len(stretch) < min_rows:
                continue
            rows = [c.frame for c in stretch]
            if up is not None:
                covered[rows, upper(scene, pair.a, pair.b, rows, up)] = True
            record = scene.proximity.record(pair, stretch)
            yield replace(record, kind=ContactKind.RESTING, source=SOURCE)


def consecutive(contacts: Sequence[FrameContact]) -> Iterator[list[FrameContact]]:
    current: list[FrameContact] = []
    for contact in contacts:
        if current and contact.frame != current[-1].frame + 1:
            yield current
            current = []
        current.append(contact)
    if current:
        yield current


def upper(scene: KinematicScene, a: int, b: int, rows: Sequence[int], up: FloatArray) -> int:
    heights = [nan_median((scene.states.position[rows, n] @ up).tolist()) for n in (a, b)]
    return b if heights[1] > heights[0] else a


def scene_planes(
    inputs: SceneInputs,
    up: FloatArray | None,
    env_runs: Sequence[tuple[slice, ...]],
    events: Sequence[Event],
    tests: Counter[int],
) -> tuple[SupportPlane | None, ...]:
    """P_n of every object: evidence from rest runs on the environment and upward impacts."""
    states = inputs.scene.states
    if up is None:
        return (None,) * states.n_objects
    evidence: dict[int, list[float]] = {n: [] for n in range(states.n_objects)}
    bodies = inputs.scene.proximity.bodies
    for n, runs in enumerate(env_runs):
        for run in runs:
            evidence[n].extend(height(bodies[t][n], up) for t in range(run.start, run.stop))
    for event in events:
        n, impact = event.index, event.impact
        change = NormalChange.along(impact.delta_v, up, float(inputs.sigma_v[n]), tests[n])
        if change.significant(inputs.alpha):
            body = bodies[nearest_row(inputs.scene, impact.time)][n]
            evidence[n].append(height(body, up, impact.position))
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


def gate(
    inputs: SceneInputs,
    choice: Choice,
    tests: Counter[int],
    planes: Sequence[SupportPlane | None],
) -> TestedChoice:
    """Significance of the change along the chosen normal: environment du = dv . up with m_n;
    opposite impacts (a) du = (dv_a - dv_b) . n with hypot(sigma_a, sigma_b) and max(m_a, m_b);
    a proximity pair (b) du = dv_a . n with m_a."""
    sigma_v, index = inputs.sigma_v, choice.event.index
    delta_v = choice.event.impact.delta_v
    if choice.other is None and planes[index] is None:
        return TestedChoice(choice, None, False)
    sigma, m = float(sigma_v[index]), tests[index]
    if choice.partner is not None:
        delta_v = delta_v - choice.partner.impact.delta_v
        sigma = math.hypot(sigma, float(sigma_v[choice.partner.index]))
        m = max(m, tests[choice.partner.index])
    change = NormalChange.along(delta_v, choice.normal, sigma, m)
    return TestedChoice(choice, change, change.significant(inputs.alpha))


def impact_record(
    scene: KinematicScene, tested: TestedChoice, planes: Sequence[SupportPlane | None]
) -> ContactRecord | None:
    choice, change = tested.choice, tested.change
    assert change is not None
    if choice.other is not None:
        record = record_of(scene, choice)
        if record is None:
            return None
        return replace(record, confidence=change.confidence, source=SOURCE)
    support = planes[choice.event.index]
    assert support is not None
    return env_impact(scene, choice.event, support, change)


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
