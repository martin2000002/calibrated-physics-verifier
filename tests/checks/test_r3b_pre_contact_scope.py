import inspect
import math
from collections.abc import Sequence
from dataclasses import replace
from itertools import combinations
from typing import Any, get_type_hints

import check_scenarios as sc
import disc_tracks
import numpy as np
import pytest

from physverify.app.plugins import PLUGIN_KINDS
from physverify.checks.variants._image_impulse import Event, FloatArray
from physverify.checks.variants.event_scopes._base import EVENT_SCOPES, ScopeLimits, SidedEvent
from physverify.checks.variants.event_scopes.pre_contact_approach import (
    PreContactApproach,
    Tracks,
)
from physverify.checks.variants.r3b_image_relspeed import RelSpeedCheck, RelSpeedParams
from physverify.extraction.centres.contour_circle import ContourCircleCentre, Params
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.provenance import GROUND_TRUTH
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.verification.runner import RowStatus, aggregate

D = "pre_contact_approach"
SCOPED = RelSpeedParams(event_scopes=(D,))
LIMITS = RelSpeedParams().scope_limits()
TRUTH = {Stage.CONTACTS.value: GROUND_TRUTH}
SPHERE, BOX = PrimitiveKind.SPHERE, PrimitiveKind.BOX
ALL = ("spheres", "onset", "approach_seen", "isolated_before", "matched")
SIDE = 0.2  # every body's extents: apparent radius 0.1 / z
FRAMES = 32
PROPERTY_CASES = 240
MASK_CASES = 200


def table(
    position: FloatArray, kinds: Sequence[PrimitiveKind], step: int = 1, size: Any = SIDE
) -> ObjectStateTable:
    """Camera-frame states of bodies "b0", "b1", ... at ``position`` (frames, bodies, 3)."""
    frames, count = position.shape[:2]
    objects = [
        sc.still(f"b{n}", kind, sc.cube(SIDE), (0.0, 1.0, 0.0), frames)
        for n, kind in enumerate(kinds)
    ]
    states = sc.build("pre_contact", objects).states
    sizes = np.broadcast_to(np.asarray(size, dtype=np.float64), (frames, count, 3)).copy()
    return replace(
        states, position=position, size=sizes, frame_index=np.asarray(states.frame_index) * step
    )


def head_on(speed: float = 0.04, frames: int = FRAMES) -> FloatArray:
    """b0 and b1 at depth 1 closing along x by 2 ``speed`` per row: d(k) = 2 - 2 speed k."""
    k = np.arange(frames, dtype=np.float64)
    position = np.zeros((frames, 2, 3))
    position[:, 0, 0], position[:, 1, 0] = -1.0 + speed * k, 1.0 - speed * k
    position[:, :, 2] = 1.0
    return position


def with_third(position: FloatArray, at: FloatArray) -> FloatArray:
    return np.concatenate([position, at[:, None, :]], axis=1)


def failed(
    states: ObjectStateTable,
    t: float,
    pair: tuple[str, str] = ("b0", "b1"),
    gates: ScopeLimits = LIMITS,
) -> tuple[str, ...]:
    return PreContactApproach().failed(states, SidedEvent(Event(*pair, t), (), ()), [], gates)


def onset(states: ObjectStateTable, gates: ScopeLimits = LIMITS) -> int | None:
    return Tracks.of(states, gates).onset(0, 1)


def test_head_on_spheres_seen_and_alone_are_in_scope() -> None:
    states = table(head_on(), [SPHERE, SPHERE])
    assert onset(states) == 22
    assert failed(states, 22.4) == ()


def test_an_onset_by_closing_speed_comes_before_the_proximity_one() -> None:
    gates = replace(LIMITS, kappa=1.0)
    assert onset(table(head_on(), [SPHERE, SPHERE]), gates) == 22
    k = np.arange(FRAMES)
    still_after = head_on()
    still_after[k >= 22, 0, 0], still_after[k >= 22, 1, 0] = -0.12, 0.12
    assert onset(table(still_after, [SPHERE, SPHERE]), gates) == 22
    halted = head_on()
    halted[k >= 21, 0, 0], halted[k >= 21, 1, 0] = -0.16, 0.16
    assert onset(table(halted, [SPHERE, SPHERE]), gates) is None


@pytest.mark.parametrize("kinds", [(BOX, SPHERE), (SPHERE, PrimitiveKind.CYLINDER)])
def test_both_bodies_must_be_spheres(kinds: tuple[PrimitiveKind, PrimitiveKind]) -> None:
    assert failed(table(head_on(), kinds), 22.0) == ("spheres",)


def test_a_pair_that_never_comes_near_fails_every_onset_predicate() -> None:
    assert failed(table(head_on(speed=0.0), [SPHERE, SPHERE]), 10.0) == ALL[1:]


@pytest.mark.parametrize(("row", "seen"), [(18, True), (19, False), (21, False), (20, False)])
def test_the_approach_rows_ending_one_gap_before_the_onset_need_centres(
    row: int, seen: bool
) -> None:
    position = head_on()
    position[row, 1] = math.nan
    states = table(position, [SPHERE, SPHERE])
    assert onset(states) == 22
    assert ("approach_seen" not in failed(states, 22.0)) is seen


def test_the_approach_rows_cannot_start_before_the_clip() -> None:
    states = table(head_on(speed=0.4, frames=8), [SPHERE, SPHERE])
    assert onset(states) == 2
    assert failed(states, 2.0) == ("approach_seen",)


@pytest.mark.parametrize(
    ("step", "row", "isolated"), [(1, 14, False), (1, 13, True), (2, 15, False), (2, 14, True)]
)
def test_no_third_body_is_near_either_one_within_the_reach_before_the_onset(
    step: int, row: int, isolated: bool
) -> None:
    position = head_on()
    third = np.tile([0.0, -1.0, 1.0], (FRAMES, 1))
    third[row] = position[row, 1]
    states = table(with_third(position, third), [SPHERE, SPHERE, BOX], step=step)
    assert onset(states) == 22
    assert ("isolated_before" not in failed(states, 22.0 * step)) is isolated


def test_a_third_body_near_after_the_onset_is_not_read() -> None:
    position = head_on()
    third = np.tile([0.0, -1.0, 1.0], (FRAMES, 1))
    third[23:] = position[23:, 0]
    assert failed(table(with_third(position, third), [SPHERE, SPHERE, SPHERE]), 22.0) == ()


@pytest.mark.parametrize(
    ("step", "t", "matched"),
    [
        (1, 30.0, True),
        (1, 14.0, True),
        (1, 30.5, False),
        (1, 13.5, False),
        (2, 59.0, True),
        (2, 59.5, False),
    ],
)
def test_the_event_matches_an_onset_within_the_reach(step: int, t: float, matched: bool) -> None:
    states = table(head_on(), [SPHERE, SPHERE], step=step)
    assert failed(states, t) == (() if matched else ("matched",))


def reference_onset(
    states: ObjectStateTable, i: int, j: int, kappa: float, half: int
) -> int | None:
    """Row-by-row restatement of the definition (the plan's tap), against the vectorized one."""

    def radius(n: int, k: int) -> float:
        size = np.asarray(states.size[max(0, k - 2 * half) : k + 1, n], dtype=np.float64)
        z = np.asarray(states.position[max(0, k - 2 * half) : k + 1, n, 2], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = 0.25 * (size[:, 0] + size[:, 1]) / z
        r = r[np.isfinite(r) & (z > 0)]
        return float(np.median(r)) if r.size else math.nan

    centres = [states.position[:, n, :2] / states.position[:, n, 2:3] for n in (i, j)]
    for c, n in zip(centres, (i, j), strict=True):
        c[
            ~(np.all(np.isfinite(states.position[:, n]), axis=1) & (states.position[:, n, 2] > 0))
        ] = math.nan
    d = np.linalg.norm(centres[0] - centres[1], axis=1)
    for k in range(d.size):
        s = radius(i, k) + radius(j, k)
        if not (np.isfinite(d[k]) and np.isfinite(s)):
            continue
        if d[k] <= kappa * s or (k >= 1 and np.isfinite(d[k - 1]) and d[k - 1] - d[k] >= d[k] - s):
            return k
    return None


def random_scene(rng: np.random.Generator) -> tuple[FloatArray, FloatArray, list[PrimitiveKind]]:
    """b0 and b1 cross in the image between rows 6 and 26, seen around the crossing; up to two
    more bodies wander; centres and sizes go missing."""
    count = int(rng.integers(2, 5))
    k = np.arange(FRAMES, dtype=np.float64)
    image = np.empty((FRAMES, count, 2))
    start, crossing, share = rng.uniform(0.3, 1.0, 2), rng.uniform(6.0, 26.0), rng.uniform(0.2, 0.8)
    closing = start.sum() / crossing
    image[:, 0, 0] = -start[0] + share * closing * k
    image[:, 1, 0] = start[1] - (1.0 - share) * closing * k
    image[:, :2, 1] = rng.normal(0.0, 0.03, (FRAMES, 2))
    drift = rng.uniform(-0.05, 0.05, (count - 2, 2))
    image[:, 2:] = rng.uniform(-1.0, 1.0, (count - 2, 2)) + drift * k[:, None, None]
    z = rng.uniform(1.5, 4.0, count) + rng.normal(0.0, 0.05, (FRAMES, count))
    position = np.concatenate([image * z[..., None], z[..., None]], axis=2)
    size = rng.uniform(0.1, 0.4, (FRAMES, count, 3))
    holes = rng.random((FRAMES, count)) < 0.08
    holes[np.abs(k - crossing) <= 1.5, :2] = False
    position[holes] = math.nan
    size[rng.random((FRAMES, count)) < 0.05] = math.nan
    kinds = [SPHERE if rng.random() < 0.8 else BOX for _ in range(count)]
    return position, size, kinds


def rewritten_after(
    position: FloatArray, size: FloatArray, row: int, rng: np.random.Generator
) -> tuple[FloatArray, FloatArray]:
    """Arbitrary rows after ``row``: other places (some on top of b0), other sizes, holes."""
    position, size = position.copy(), size.copy()
    late = slice(row + 1, None)
    position[late] = rng.uniform(-1, 1, position[late].shape) + np.array([0.0, 0.0, 2.5])
    position[late, 1:] = np.where(
        rng.random(position[late, 1:, :1].shape) < 0.3, position[late, :1], position[late, 1:]
    )
    size[late] = rng.uniform(0.05, 0.8, size[late].shape)
    position[late][rng.random(position[late].shape[:2]) < 0.2] = math.nan
    return position, size


@pytest.mark.parametrize("seed", range(PROPERTY_CASES))
def test_the_decision_depends_on_the_rows_up_to_the_onset_only(seed: int) -> None:
    rng = np.random.default_rng(seed)
    position, size, kinds = random_scene(rng)
    states = table(position, kinds, size=size)
    tracks = Tracks.of(states, LIMITS)
    first = tracks.onset(0, 1)
    assert first is not None
    assert first == reference_onset(states, 0, 1, LIMITS.kappa, LIMITS.radius_rows)
    t = first + float(rng.uniform(-12.0, 12.0))
    decided = failed(states, t)
    onsets = {(i, j): tracks.onset(i, j) for i, j in combinations(range(len(kinds)), 2)}
    late, late_size = rewritten_after(position, size, first, rng)
    mutated = table(late, kinds, size=late_size)
    assert Tracks.of(mutated, LIMITS).onset(0, 1) == first
    assert failed(mutated, t) == decided
    mutated_tracks = Tracks.of(mutated, LIMITS)
    for (i, j), k in onsets.items():
        if k is not None and k <= first:
            assert mutated_tracks.onset(i, j) == k


def test_the_scope_reads_the_causal_centre_where_the_lift_gives_one() -> None:
    position = head_on()
    apart = table(head_on(speed=0.0), [SPHERE, SPHERE])
    assert onset(replace(apart, causal_centre=np.full((FRAMES, 2, 2), math.nan))) is None
    states = replace(apart, causal_centre=position[..., :2] / position[..., 2:])
    assert onset(states) == 22
    assert failed(states, 22.0) == ()


def from_masks(masks: np.ndarray, causal: bool) -> ObjectStateTable:
    """States of sphere tracks at depth 1 in pixel units (f = 1, no offset): the contour centre
    as position, its causal centre when ``causal``, and each row's own sqrt(A / pi) as radius."""
    count = masks.shape[1]
    centres = ContourCircleCentre(Params()).image_centres(masks, (SPHERE,) * count)
    assert centres is not None and centres.causal_uv is not None
    position = np.concatenate([centres.uv, np.ones((*centres.uv.shape[:2], 1))], axis=2)
    area = masks.sum(axis=(2, 3)).astype(np.float64)
    radius = np.where(area > 0, np.sqrt(area / math.pi), math.nan)
    states = table(position, [SPHERE] * count, size=np.repeat(2.0 * radius[..., None], 3, axis=2))
    return replace(states, causal_centre=centres.causal_uv if causal else None)


def masks_hold_the_prefix(seed: int, causal: bool) -> bool:
    """Whether rewriting the masks after k* leaves k*, every onset up to it, the decision and the
    scope's inputs up to k* unchanged."""
    rng = np.random.default_rng(seed)
    masks = disc_tracks.approach(rng)
    states = from_masks(masks, causal)
    tracks = Tracks.of(states, LIMITS)
    first = tracks.onset(0, 1)
    assert first is not None
    t = first + float(rng.uniform(-12.0, 12.0))
    mutated = from_masks(disc_tracks.rewritten_after(masks, first, rng), causal)
    after = Tracks.of(mutated, LIMITS)
    head = slice(0, first + 1)
    inputs = all(
        np.array_equal(a[head], b[head], equal_nan=True)
        for a, b in zip(
            (*tracks.centres, *tracks.radii), (*after.centres, *after.radii), strict=True
        )
    )
    onsets = all(
        after.onset(i, j) == k
        for i, j in combinations(range(masks.shape[1]), 2)
        if (k := tracks.onset(i, j)) is not None and k <= first
    )
    return (
        inputs and onsets and after.onset(0, 1) == first and failed(mutated, t) == failed(states, t)
    )


@pytest.mark.parametrize("seed", range(MASK_CASES))
def test_masks_rewritten_after_the_onset_leave_the_decision_and_its_inputs(seed: int) -> None:
    assert masks_hold_the_prefix(seed, causal=True)


def test_the_contour_centre_without_its_causal_one_breaks_the_property() -> None:
    assert not all(masks_hold_the_prefix(seed, causal=False) for seed in range(20))


def test_the_property_cases_span_both_decisions() -> None:
    inside = 0
    for seed in range(PROPERTY_CASES):
        position, size, kinds = random_scene(np.random.default_rng(seed))
        states = table(position, kinds, size=size)
        k = Tracks.of(states, LIMITS).onset(0, 1)
        assert k is not None
        inside += failed(states, float(k)) == ()
    assert 0.2 * PROPERTY_CASES < inside < 0.8 * PROPERTY_CASES


def evaluated(params: RelSpeedParams, case: sc.Case, truth: bool) -> list[CheckResult]:
    level = InputLevel.ORACLE if truth else InputLevel.PERCEIVED
    return RelSpeedCheck(params).evaluate(case.inputs(level, TRUTH if truth else {}))


def outputs(results: Sequence[CheckResult]) -> list[tuple[Any, ...]]:
    return [
        (r.subject, r.magnitude, r.sigma, r.verdict, r.abstain_reason, r.details) for r in results
    ]


@pytest.mark.parametrize("truth", [False, True], ids=["perceived", "truth"])
def test_an_event_in_scope_is_unchanged(truth: bool) -> None:
    case = sc.collision(gain=0.2)
    plain = evaluated(RelSpeedParams(), case, truth)
    assert len(plain) == 1 and plain[0].magnitude is not None
    assert outputs(evaluated(SCOPED, case, truth)) == outputs(plain)


@pytest.mark.parametrize("truth", [False, True], ids=["perceived", "truth"])
def test_an_event_outside_the_scope_is_not_applicable_and_keeps_its_details(truth: bool) -> None:
    case = sc.collision(gain=0.2, kind=BOX)
    (plain,) = evaluated(RelSpeedParams(), case, truth)
    (scoped,) = evaluated(SCOPED, case, truth)
    assert plain.magnitude is not None
    assert (
        scoped.verdict is Verdict.ABSTAIN and scoped.abstain_reason is AbstainReason.NOT_APPLICABLE
    )
    assert scoped.details == {
        **plain.details,
        "not_applicable": D,
        "not_applicable_predicates": ["spheres"],
    }
    assert aggregate(scoped.clip_id, RelSpeedCheck(SCOPED), [scoped]).status is RowStatus.NO_EVENT


def test_an_abstention_in_scope_keeps_its_reason_and_counts_as_abstained() -> None:
    case = sc.collision(gain=-1.0)
    plain = evaluated(RelSpeedParams(), case, truth=True)
    assert [r.abstain_reason for r in plain] == [AbstainReason.ILL_CONDITIONED]
    scoped = evaluated(SCOPED, case, truth=True)
    assert outputs(scoped) == outputs(plain)
    assert aggregate(case.clip.clip_id, RelSpeedCheck(SCOPED), scoped).status is RowStatus.ABSTAINED


def test_the_scope_has_the_event_scope_signature_and_is_registered() -> None:
    inputs = {
        "states": ObjectStateTable,
        "event": SidedEvent,
        "others": Sequence[Event],
        "gates": ScopeLimits,
    }
    for method, returned in (("applies", bool), ("failed", tuple[str, ...])):
        assert list(inspect.signature(getattr(PreContactApproach, method)).parameters) == [
            "self",
            *inputs,
        ]
        assert get_type_hints(getattr(PreContactApproach, method)) == {**inputs, "return": returned}
    assert PLUGIN_KINDS[EVENT_SCOPES.name]()[D] is PreContactApproach
    assert [type(s) for s in RelSpeedCheck(SCOPED).event_scopes] == [PreContactApproach]
    assert RelSpeedCheck(RelSpeedParams()).event_scopes == ()
