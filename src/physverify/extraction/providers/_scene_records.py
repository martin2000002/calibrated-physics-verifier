"""Contact records of the kinematic scene provider; not a plugin.

Existence is kinematic. RESTING: rest runs of both objects of a proximity pair (pair record), or
rest rows of an object not resting on another one, on its scene plane (environment record).
IMPACT: a velocity kink whose shape explains its window best, attributed to a surface as in
``_impact_attribution.py`` and kept when its change along the surface normal is significant.
Environment records place their points midway between the body and the scene plane, so that
``support_plane`` returns the scene plane.
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
from physverify.physics.geometry import Solid
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
    the lift's height error and the test settings."""

    scene: KinematicScene
    intervals: tuple[Interval, ...]
    sigma_v: FloatArray
    height_error: HeightError
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
    records: tuple[ContactRecord, ...]
    tested: tuple[TestedChoice, ...]
    planes: tuple[SupportPlane | None, ...]
    rest: BoolArray


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
    resting = [
        r
        for n, runs in enumerate(env_runs)
        if (support := planes[n]) is not None
        for run in runs
        if (r := env_resting(inputs, n, run, support))
    ]
    records = sorted(
        [*pairs, *impacts, *resting], key=lambda r: (r.object_a, r.object_b, r.t_start)
    )
    return SceneOutcome(tuple(records), tuple(tested), planes, rest)


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
    """One point b - (g / 2) up with g = up . b - c, b the body's lowest point at the impact."""
    up, impact = support.plane.normal, event.impact
    row = nearest_row(scene, impact.time)
    bottom = facing_point(scene, event.index, row, impact.position, up)
    if bottom is None:
        return None
    gap = float(up @ bottom) - support.height
    record = make_record(scene, (event.index, None), impact.time, up, bottom - gap / 2.0 * up, row)
    return replace(
        record,
        point_distances=np.array([gap]),
        confidence=change.confidence,
        source=SOURCE,
    )


def env_resting(
    inputs: SceneInputs, n: int, run: slice, support: SupportPlane
) -> ContactRecord | None:
    """RESTING on P_n over ``run``: the patch of the body moved onto the plane at the central
    row, projected on the plane and lifted by g / 2; the gap is only a confidence."""
    scene = inputs.scene
    states, proximity = scene.states, scene.proximity
    rows = [t for t in range(run.start, run.stop) if solid(proximity.bodies[t][n]) is not None]
    if not rows:
        return None
    gaps = [support_gap(proximity.bodies[t][n], support) for t in rows]
    middle = len(rows) // 2
    placed = proximity.bodies[rows[middle]][n]
    body = solid(placed)
    assert placed is not None and body is not None
    length = float(proximity.lengths[n])
    patch = plane_patch(body, support, gaps[middle], scene.tol_rel * length)
    count = patch.shape[0]
    up = support.plane.normal
    spread = math.hypot(inputs.height_error.sigma(length), support.sigma)
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
        point_distances=np.full(count, gaps[middle]),
        orientation_assumed=placed.assumed,
        max_penetration=max(0.0, -min(gaps)),
        confidence=float(2.0 * norm.sf(abs(float(np.median(gaps))) / spread)),
        source=SOURCE,
    )


def plane_patch(body: Solid, support: SupportPlane, gap: float, tol: float) -> FloatArray:
    """Contact patch of the body moved by -gap along up, on the plane, lifted by gap / 2."""
    plane, up = support.plane, support.plane.normal
    touching = moved(body, body.pose.position - gap * up)
    manifold = contact_manifold(touching, plane, tol)
    points = (
        signed_distance(touching, plane).point_a[None, :] if manifold is None else manifold.points
    )
    on_plane = points - (points @ up - plane.offset)[:, None] * up
    lifted: FloatArray = on_plane + gap / 2.0 * up
    return lifted
