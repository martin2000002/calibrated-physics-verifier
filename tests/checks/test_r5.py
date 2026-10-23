import math
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks._base import Check, CheckInputs
from physverify.checks.r5a_fall_shape import PLUGINS as R5A_PLUGINS
from physverify.checks.r5a_fall_shape import FallShapeCheck, FallShapeParams
from physverify.checks.r5b_gravity_value import PLUGINS as R5B_PLUGINS
from physverify.checks.r5b_gravity_value import GravityValueCheck, GravityValueParams
from physverify.physics.measures.gravity_fit import accel_inconsistency, fall_fit
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.constants import STANDARD_GRAVITY
from physverify.schema.scale import ScaleSource
from physverify.schema.states import PrimitiveKind

G = STANDARD_GRAVITY


def inputs(case: sc.Case) -> CheckInputs:
    return CheckInputs(
        clip=case.clip,
        states=case.states,
        contacts=case.contacts,
        scale=case.scale,
        input_level=InputLevel.ORACLE,
        provenance={},
        scene=case.scene,
    )


def run(check: Check, case: sc.Case) -> list[CheckResult]:
    results = check.evaluate(inputs(case))
    native = set(case.states.frame_index.tolist())
    for result in results:
        if result.frame_span is not None:
            assert set(result.frame_span) <= native
    return results


def shape(case: sc.Case) -> CheckResult:
    results = run(FallShapeCheck(FallShapeParams()), case)
    assert len(results) == 1
    return results[0]


def gravity(case: sc.Case) -> CheckResult:
    results = run(GravityValueCheck(GravityValueParams()), case)
    assert len(results) == 1
    return results[0]


def halves(case: sc.Case, change_frame: int) -> float:
    """accel_inconsistency of the fits on the rows before and after the native change frame."""
    states = case.states
    late = states.frame_index >= change_frame
    fits = [fall_fit(states.t[rows], states.position[rows, 0]) for rows in (~late, late)]
    return accel_inconsistency([fit.a for fit in fits])


def two_balls(g_j: float) -> sc.Case:
    t = sc.times()
    balls = []
    for name, g, x in (("ball_i", G, -0.5), ("ball_j", g_j, 0.5)):
        start, v0, a = np.array([x, 0.0, 1.0]), np.array([0.5, 0.0, 3.0]), g * sc.DOWN
        p, v, acc = sc.piecewise_ballistic(t, 0.0, start, (v0, a), (v0, a))
        balls.append(sc.SceneObject(name, PrimitiveKind.SPHERE, sc.cube(0.2), p, v, acc))
    return sc.build("two_balls", balls)


def test_plugins_satisfy_the_check_protocol() -> None:
    assert {cls.check_id for cls in (*R5A_PLUGINS, *R5B_PLUGINS)} == {"R5a", "R5b"}
    for cls in (*R5A_PLUGINS, *R5B_PLUGINS):
        assert isinstance(cls(cls.Params()), Check)


# R5a.


@pytest.mark.parametrize("pitch", [0.0, 25.0])
def test_correct_parabola_has_one_acceleration(pitch: float) -> None:
    result = shape(sc.parabola(pitch_deg=pitch))
    assert result.magnitude is not None
    assert result.magnitude <= 1e-9
    assert result.sigma == 0.0
    assert result.frame_span == (0, sc.FRAMES - 1)
    assert result.details["lateral_accel_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert result.details["fall_residual_ratio"] <= 1e-9


@pytest.mark.parametrize("step", [1, 2, 3])
def test_gravity_change_is_split_at_the_change_frame(step: int) -> None:
    native = sc.parabola(g_after=1.3 * G, change_frame=18)
    case = sc.strided(native, step, quantize=True)
    result = shape(case)
    assert result.magnitude == pytest.approx(halves(native, 18), abs=1e-9)
    assert result.magnitude == pytest.approx(halves(case, 18), abs=1e-9)
    assert result.details["mode"] == "split"
    assert result.details["split_frame"] == 18
    fi = case.states.frame_index
    assert result.frame_span == (int(fi[0]), int(fi[-1]))


def test_gravity_change_between_frames_splits_at_the_next_row() -> None:
    case = sc.parabola(g_after=1.3 * G, change_frame=17.5)
    result = shape(case)
    assert result.details["split_frame"] == 18
    assert result.magnitude == pytest.approx(halves(case, 18), abs=1e-9)


@pytest.mark.parametrize("step", [1, 2])
def test_teleport_in_flight_does_not_change_the_shape(step: int) -> None:
    case = sc.strided(sc.teleport(0.3, frame=16, flight=True), step, quantize=True)
    result = shape(case)
    assert result.magnitude is not None
    assert result.magnitude <= 1e-6


def test_consecutive_flights_of_a_bounce_agree() -> None:
    result = shape(sc.bounce())
    assert result.magnitude is not None
    assert result.magnitude <= 1e-9


def test_objects_with_different_gravity_are_compared() -> None:
    results = run(FallShapeCheck(FallShapeParams()), two_balls(1.2 * G))
    by_subject = {r.subject: r for r in results}
    assert set(by_subject) == {("ball_i",), ("ball_j",), ("ball_i", "ball_j")}
    pair = by_subject[("ball_i", "ball_j")]
    assert pair.magnitude == pytest.approx(0.2 / 1.1, abs=1e-9)
    assert pair.details["mode"] == "pair"


def test_unknown_gravity_direction_does_not_abstain() -> None:
    case = sc.parabola()
    case = replace(case, states=replace(case.states, gravity_dir=np.full(3, math.nan)))
    result = shape(case)
    assert result.verdict is Verdict.UNSCORED
    assert result.details["lateral_accel_ratio"] is None


def test_short_flight_abstains_and_no_flight_is_silent() -> None:
    result = shape(sc.parabola(frames=5))
    assert result.abstain_reason is AbstainReason.TOO_FEW_FRAMES
    assert result.frame_span == (0, 4)
    assert run(FallShapeCheck(FallShapeParams()), sc.stack()) == []


def test_flight_without_gravity_is_not_a_fall() -> None:
    assert run(FallShapeCheck(FallShapeParams()), sc.collision()) == []


def hovering_ball(sigma: float = 0.01) -> sc.Case:
    """A sphere held in the air without contacts, seen with position noise and no velocities."""
    ball = sc.still("ball", PrimitiveKind.SPHERE, sc.cube(2 * sc.BALL_RADIUS), [0.0, 0.0, 1.0])
    case = sc.build("hover", [ball])
    states = case.states
    noise = np.random.default_rng(0).normal(0.0, sigma, states.position.shape)
    noisy = replace(
        states,
        position=states.position + noise,
        position_sigma=np.full(states.position.shape, sigma),
        velocity=np.full(states.velocity.shape, math.nan),
    )
    return replace(case, states=noisy)


@pytest.mark.parametrize(
    ("check", "params"),
    [(FallShapeCheck, FallShapeParams), (GravityValueCheck, GravityValueParams)],
)
def test_noisy_hovering_is_not_a_fall(check: type, params: type) -> None:
    case = hovering_ball()
    assert run(check(params()), case) == []
    unfiltered = params(min_travel_lengths=0.0, travel_sigmas=0.0)
    assert run(check(unfiltered), case) != []


def drifting_ball(amplitude: float, sigma: float = 0.01) -> sc.Case:
    """``hovering_ball`` plus a slow quadratic drift along x of ``amplitude`` over the clip."""
    case = hovering_ball(sigma)
    t = case.states.t
    drift = amplitude * ((t - t[0]) / (t[-1] - t[0])) ** 2
    position = case.states.position.copy()
    position[:, 0, 0] += drift
    return replace(case, states=replace(case.states, position=position))


def declared_far_too_large(length_sigmas: float = 6.0, noise_lengths: float = 0.01) -> sc.Case:
    """A real fall whose declared position sigma is ``length_sigmas`` body lengths while its
    actual white noise is ``noise_lengths`` body lengths (a defect a perception iteration found
    in the extractor diagnosis)."""
    case = sc.parabola()
    length = 2 * sc.BALL_RADIUS
    noise = np.random.default_rng(3).normal(0.0, noise_lengths * length, case.states.position.shape)
    states = replace(
        case.states,
        position=case.states.position + noise,
        position_sigma=np.full(case.states.position.shape, length_sigmas * length),
    )
    return replace(case, states=states)


CHECKS = [(FallShapeCheck, FallShapeParams), (GravityValueCheck, GravityValueParams)]


@pytest.mark.parametrize(("check", "params"), CHECKS)
def test_a_fall_hidden_by_a_declared_absolute_sigma_is_found_on_the_track(
    check: type, params: type
) -> None:
    case = declared_far_too_large()
    assert run(check(params(travel_noise="declared")), case) == []
    assert run(check(params(travel_noise="track")), case) != []


@pytest.mark.parametrize(("check", "params"), CHECKS)
def test_t9b_slow_drift_below_half_a_length_is_not_a_fall(check: type, params: type) -> None:
    """The third difference does not see a drift that is quadratic over the stencil: only the
    length floor stops it, as the run without that floor shows."""
    case = drifting_ball(0.45 * 2 * sc.BALL_RADIUS, sigma=0.002)
    assert run(check(params(travel_noise="track")), case) == []
    no_length = params(travel_noise="track", min_travel_lengths=0.0)
    assert run(check(no_length), case) != []


@pytest.mark.parametrize(("check", "params"), CHECKS)
@pytest.mark.parametrize("case", [sc.parabola(), sc.bounce(), sc.parabola(g_after=1.3 * G)])
def test_exact_positions_keep_the_length_floor(check: type, params: type, case: sc.Case) -> None:
    assert bool(np.all(case.states.position_sigma == 0.0))
    track = run(check(params(travel_noise="track")), case)
    declared = run(check(params(travel_noise="declared")), case)
    assert track
    assert track == declared


@pytest.mark.parametrize(("check", "params"), CHECKS)
def test_without_a_known_sigma_the_track_noise_comes_before_the_length_floor(
    check: type, params: type
) -> None:
    """No finite sigma, enough stencils -> the track noise sets the floor."""
    case = hovering_ball()
    unknown = replace(
        case,
        states=replace(case.states, position_sigma=np.full(case.states.position.shape, math.nan)),
    )
    assert run(check(params(min_travel_lengths=0.0, travel_noise="track")), unknown) == []
    assert run(check(params(min_travel_lengths=0.0, travel_noise="declared")), unknown) != []


@pytest.mark.parametrize(("check", "params"), CHECKS)
def test_too_few_stencils_fall_back_to_the_declared_sigma(check: type, params: type) -> None:
    case = declared_far_too_large()
    rows = case.states.n_frames - 3
    assert run(check(params(travel_noise="track", min_noise_stencils=rows + 1)), case) == []
    assert run(check(params(travel_noise="track", min_noise_stencils=rows)), case) != []


def test_shape_sigma_follows_position_noise() -> None:
    case = sc.with_sigma(sc.parabola(g_after=1.3 * G, change_frame=18), position=0.002)
    result = shape(case)
    assert result.sigma > 0.0
    nan = sc.with_sigma(sc.parabola(), position=math.nan)
    assert math.isnan(shape(nan).sigma)


# R5b.


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize(("g", "expected"), [(G, 0.0), (1.2 * G, 0.2)])
def test_gravity_value_does_not_depend_on_the_row_step(
    step: int, g: float, expected: float
) -> None:
    case = sc.strided(sc.parabola(g=g), step, quantize=True)
    result = gravity(case)
    assert result.magnitude == pytest.approx(expected, abs=1e-9)
    assert result.subject == ()
    assert result.sigma == 0.0
    fi = case.states.frame_index
    assert result.frame_span == (int(fi[0]), int(fi[-1]))
    assert result.details["scale_source"] == "ground_truth"
    assert result.details["row_step"] == step


def test_gravity_value_takes_the_median_flight() -> None:
    result = gravity(sc.bounce())
    assert result.magnitude == pytest.approx(0.0, abs=1e-9)
    assert result.details["used_flights"] == 2


def test_gravity_value_applies_the_scale() -> None:
    case = replace(sc.parabola(), scale=sc.oracle_scale(meters_per_unit=1.1))
    assert gravity(case).magnitude == pytest.approx(0.1, abs=1e-9)


@pytest.mark.parametrize(
    ("scale", "reason"),
    [
        (None, AbstainReason.SCALE_UNAVAILABLE),
        (sc.oracle_scale(source=ScaleSource.FREE_FALL), AbstainReason.SCALE_FROM_GRAVITY),
    ],
)
def test_gravity_value_needs_an_independent_scale(scale: object, reason: AbstainReason) -> None:
    result = gravity(replace(sc.parabola(), scale=scale))
    assert result.abstain_reason is reason
    assert result.frame_span == (0, sc.FRAMES - 1)


def test_gravity_value_short_flight_abstains() -> None:
    result = gravity(sc.parabola(frames=3))
    assert result.abstain_reason is AbstainReason.TOO_FEW_FRAMES


def test_gravity_value_is_for_spheres_only() -> None:
    assert run(GravityValueCheck(GravityValueParams()), sc.bounce(kind=PrimitiveKind.BOX)) == []


def test_gravity_sigma_adds_the_scale_sigma_in_quadrature() -> None:
    case = replace(sc.parabola(g=1.2 * G), scale=sc.oracle_scale(rel_sigma=0.1))
    assert gravity(case).sigma == pytest.approx(1.2 * 0.1, rel=1e-9)
    nan = sc.with_sigma(sc.parabola(), position=math.nan)
    assert math.isnan(gravity(nan).sigma)


@pytest.mark.parametrize("g_rel", [0.0, 0.5])
def test_gravity_value_scores_weak_gravity(g_rel: float) -> None:
    assert gravity(sc.parabola(g=g_rel * G)).magnitude == pytest.approx(abs(g_rel - 1.0), abs=1e-9)
