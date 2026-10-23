import math
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest
from reference.r4_jump_v1 import ContinuityCheck

from physverify.checks._base import Check
from physverify.checks._params import CommonParams
from physverify.checks._peaks import first_max
from physverify.checks.r4_missing_permanence import PLUGINS as MISSING_PLUGINS
from physverify.checks.r4_missing_permanence import PermanenceCheck, PermanenceParams
from physverify.checks.r4_size_constancy import PLUGINS as SIZE_PLUGINS
from physverify.checks.r4_size_constancy import SizeConstancyCheck, SizeParams
from physverify.physics.finite_diff import third_difference
from physverify.physics.measures.object_integrity import size_change_ratio
from physverify.schema.checks import AbstainReason, CheckResult, Verdict
from physverify.schema.constants import STANDARD_GRAVITY

BALL_LENGTH = 2 * sc.BALL_RADIUS


def run_one(check: Check, case: sc.Case) -> CheckResult:
    results = check.evaluate(sc.bare_inputs(case))
    assert len(results) == 1
    result = results[0]
    if result.frame_span is not None:
        native = set(case.states.frame_index.tolist())
        assert set(result.frame_span) <= native
    return result


def missing(case: sc.Case) -> CheckResult:
    return run_one(PermanenceCheck(PermanenceParams()), case)


def size(case: sc.Case) -> CheckResult:
    return run_one(SizeConstancyCheck(SizeParams(reference="self")), case)


def jump(case: sc.Case) -> CheckResult:
    return run_one(ContinuityCheck(CommonParams()), case)


def test_plugins_satisfy_the_check_protocol() -> None:
    ids = {cls.check_id for cls in (*MISSING_PLUGINS, *SIZE_PLUGINS)}
    assert ids == {"R4_missing", "R4_size"}
    for cls in (*MISSING_PLUGINS, *SIZE_PLUGINS, ContinuityCheck):
        assert isinstance(cls(cls.Params()), Check)


# R4_missing.


@pytest.mark.parametrize("pitch", [0.0, 25.0])
def test_disappearance_counts_the_unexplained_frames(pitch: float) -> None:
    result = missing(sc.disappearance(4, 7, pitch_deg=pitch))
    assert result.magnitude == 4
    assert result.sigma == 0.0
    assert result.frame_span == (4, 7)
    assert result.details["missing_rows"] == 4
    assert result.verdict is Verdict.UNSCORED


def test_occluded_frames_do_not_count() -> None:
    case = sc.with_values(sc.disappearance(4, 7), "occlusion", "ball", range(4, 8), 0.95)
    result = missing(case)
    assert result.magnitude == 0
    assert result.frame_span is None


def test_unknown_occlusion_abstains() -> None:
    case = sc.with_values(sc.disappearance(4, 7), "occlusion", "ball", range(4, 8), math.nan)
    result = missing(case)
    assert result.verdict is Verdict.ABSTAIN
    assert result.abstain_reason is AbstainReason.OCCLUDED
    assert result.frame_span == (4, 7)


def test_missing_sigma_is_unknown_for_perceived_positions_in_the_old_model() -> None:
    old = PermanenceCheck(PermanenceParams(sigma_model="unknown"))
    result = run_one(old, sc.with_sigma(sc.disappearance(4, 7), position=0.01))
    assert math.isnan(result.sigma)


def perceived_gaps(*gaps: tuple[int, int]) -> sc.Case:
    case = sc.with_sigma(sc.rolling_ball(clip_id="gaps"), position=0.01)
    for first, last in gaps:
        case = sc.with_values(case, "visible", "ball", range(first, last + 1), False)
    return case


@pytest.mark.parametrize(
    ("gaps", "runs"), [((), 1), (((4, 7),), 1), (((2, 3), (8, 8), (12, 15)), 3)]
)
def test_perceived_sigma_has_two_uncertain_edges_per_run(
    gaps: tuple[tuple[int, int], ...], runs: int
) -> None:
    result = missing(perceived_gaps(*gaps))
    assert result.magnitude == sum(last - first + 1 for first, last in gaps)
    assert result.sigma == pytest.approx(math.sqrt(2.0 * runs))
    assert result.details["sigma_estimated"] is True


def test_perceived_sigma_scales_with_the_row_step() -> None:
    case = sc.strided(sc.with_sigma(sc.disappearance(4, 7), position=0.01), 2, quantize=True)
    assert missing(case).sigma == pytest.approx(2.0 * math.sqrt(2.0))


def test_exact_positions_keep_sigma_zero() -> None:
    result = missing(sc.disappearance(4, 7))
    assert result.sigma == 0.0
    assert "sigma_estimated" not in result.details


def test_strided_disappearance_counts_native_frames() -> None:
    result = missing(sc.strided(sc.disappearance(4, 7), 2, quantize=True))
    assert result.magnitude == 4
    assert result.details["missing_rows"] == 2
    assert result.details["row_step"] == 2
    assert result.frame_span == (4, 6)


# R4_size v1 (reference "self"; v2 in test_r4_size_relative.py).


@pytest.mark.parametrize("step", [1, 2, 3])
def test_size_change_is_measured_against_the_median(step: int) -> None:
    case = sc.strided(sc.size_change(1.3, frame=20), step, quantize=True)
    result = size(case)
    assert result.magnitude == pytest.approx(0.3, abs=1e-9)
    first_grown = int(case.states.frame_index[case.states.frame_index >= 20][0])
    assert result.frame_span == (first_grown, first_grown)
    assert result.sigma == 0.0


def test_constant_size_gives_zero() -> None:
    result = size(sc.rolling_ball())
    assert result.magnitude == pytest.approx(0.0, abs=1e-12)


def test_size_sigma_follows_the_extent_sigma() -> None:
    result = size(sc.with_sigma(sc.size_change(1.3, frame=20), size=0.002))
    rel_length = math.sqrt(3 * (0.002 / (1.3 * BALL_LENGTH)) ** 2) / 3
    assert result.sigma == pytest.approx(1.3 * math.sqrt(2) * rel_length, rel=1e-9)


def test_size_with_too_few_known_rows_abstains() -> None:
    case = sc.with_values(sc.rolling_ball(), "size", "ball", range(2, sc.FRAMES), math.nan)
    result = size(case)
    assert result.abstain_reason is AbstainReason.TOO_FEW_FRAMES
    assert result.frame_span is None


def size_glitch(frames: range, factor: float = 0.5) -> sc.Case:
    case = sc.rolling_ball(clip_id="size_glitch")
    shrunk = case.states.size[np.isin(case.states.frame_index, list(frames)), 0] * factor
    return sc.with_values(case, "size", "ball", frames, shrunk)


def test_size_glitch_is_measured_when_the_rows_are_whole() -> None:
    result = size(sc.with_values(size_glitch(range(10, 12)), "occlusion", "ball", [10, 11], 0.5))
    assert result.magnitude == pytest.approx(0.5, abs=1e-9)
    assert result.details["excluded_rows"] == 0


def test_occluded_and_out_of_frame_rows_are_left_out() -> None:
    case = sc.with_values(size_glitch(range(10, 13)), "occlusion", "ball", [10, 11], 0.95)
    case = sc.with_values(case, "in_frame", "ball", [12], False)
    result = size(case)
    assert result.verdict is Verdict.UNSCORED
    assert result.magnitude == pytest.approx(0.0, abs=1e-12)
    assert result.details["excluded_rows"] == 3
    assert result.details["size_rows"] == sc.FRAMES - 3


def test_unknown_occlusion_keeps_the_perceived_result() -> None:
    """Perceived tables carry NaN occlusion and unseen rows as in_frame False with NaN size: the
    result equals the unfiltered measure over every known row (the behaviour before the filter)."""
    unseen = range(3, 6)
    case = sc.with_sigma(size_glitch(range(20, 22)), size=0.002)
    case = sc.with_values(case, "occlusion", "ball", range(sc.FRAMES), math.nan)
    case = sc.with_values(case, "in_frame", "ball", unseen, False)
    case = sc.with_values(case, "size", "ball", unseen, math.nan)
    result = size(case)
    known = sc.with_values(case, "occlusion", "ball", range(sc.FRAMES), 0.0)
    reference = size(sc.with_values(known, "in_frame", "ball", unseen, True))
    assert result.magnitude == size_change_ratio(case.states.size[:, 0])
    assert result.magnitude == pytest.approx(0.5, abs=1e-9)
    assert (result.verdict, result.magnitude, result.sigma, result.frame_span) == (
        reference.verdict,
        reference.magnitude,
        reference.sigma,
        reference.frame_span,
    )
    assert result.details["size_rows"] == sc.FRAMES - len(unseen)


@pytest.mark.parametrize("whole", [2, 3])
def test_size_abstains_only_below_min_size_frames(whole: int) -> None:
    hidden = range(whole, sc.FRAMES)
    result = size(sc.with_values(sc.rolling_ball(), "occlusion", "ball", hidden, 0.95))
    if whole < SizeParams().min_size_frames:
        assert result.abstain_reason is AbstainReason.TOO_FEW_FRAMES
        assert result.details["excluded_rows"] == len(hidden)
    else:
        assert result.verdict is Verdict.UNSCORED
        assert result.details["size_rows"] == whole


# R4_jump v1 (ContinuityCheck, out of the registry; v2 in test_r4_jump_noise.py).


@pytest.mark.parametrize("pitch", [0.0, 25.0])
def test_teleport_at_rest_is_detected(pitch: float) -> None:
    result = jump(sc.teleport(0.3, frame=16, pitch_deg=pitch))
    assert result.magnitude == pytest.approx(0.3 / BALL_LENGTH, abs=1e-9)
    assert result.frame_span == (14, 17)
    assert result.details["row"] == 15
    assert result.sigma == 0.0


@pytest.mark.parametrize("step", [2, 3])
def test_strided_teleport_keeps_magnitude_and_native_span(step: int) -> None:
    native = jump(sc.teleport(0.3, frame=18))
    case = sc.strided(sc.teleport(0.3, frame=18), step, quantize=True)
    result = jump(case)
    fi = case.states.frame_index
    k = int(np.flatnonzero(fi >= 18)[0]) - 1
    assert result.magnitude == pytest.approx(native.magnitude, abs=1e-9)
    assert result.frame_span == (int(fi[k - 1]), int(fi[k + 2]))


@pytest.mark.parametrize("pitch", [0.0, 25.0])
def test_teleport_in_flight_is_the_step_over_the_length(pitch: float) -> None:
    result = jump(sc.teleport(0.3, frame=16, flight=True, pitch_deg=pitch))
    assert result.magnitude == pytest.approx(0.3 / BALL_LENGTH, rel=1e-9)
    assert result.details["row"] == 15


@pytest.mark.parametrize("g", [STANDARD_GRAVITY, 3.0, 25.0])
@pytest.mark.parametrize("pitch", [0.0, 25.0])
def test_clean_flight_gives_zero_without_scale(g: float, pitch: float) -> None:
    case = sc.parabola(g=g, pitch_deg=pitch)
    for scale in (case.scale, None):
        result = jump(replace(case, scale=scale))
        assert result.verdict is not Verdict.ABSTAIN
        assert result.magnitude is not None
        assert result.magnitude < 1e-6
        assert result.sigma == 0.0


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("offset", [0.0, 0.4])
@pytest.mark.parametrize("quantize", [False, True])
def test_bounce_impact_is_excluded(step: int, offset: float, quantize: bool) -> None:
    """Quantized contacts (a sub-frame impact snapped to one frame) still exclude the stencil."""
    case = sc.strided(sc.bounce(impact_frame=sc.EVENT_FRAME + offset), step, quantize=quantize)
    result = jump(case)
    assert result.details["excluded_rows"] > 0
    assert result.magnitude is not None
    assert result.magnitude < 1e-6


def test_jump_without_four_known_rows_abstains() -> None:
    hidden = range(1, sc.FRAMES, 2)
    case = sc.with_values(sc.rolling_ball(), "position", "ball", hidden, math.nan)
    result = jump(case)
    assert result.abstain_reason is AbstainReason.TOO_FEW_FRAMES


def test_jump_sigma_projects_the_position_sigma_on_the_jump() -> None:
    case = sc.teleport(0.3, frame=16)
    sigma = np.zeros_like(case.states.position_sigma)
    sigma[..., 0], sigma[..., 1], sigma[..., 2] = 0.01, 0.02, 0.04
    case = replace(case, states=replace(case.states, position_sigma=sigma))
    result = jump(case)
    along = np.abs(case.view.vector(np.array([0.0, 1.0, 0.0])))
    expected = math.sqrt(5) * float(np.linalg.norm(along * sigma[0, 0])) / BALL_LENGTH
    assert result.sigma == pytest.approx(expected, rel=1e-9)


def test_jump_requires_contacts() -> None:
    case = replace(sc.rolling_ball(), contacts=None)
    with pytest.raises(ValueError, match="requires contacts"):
        ContinuityCheck(CommonParams()).evaluate(sc.bare_inputs(case))


def test_first_max_prefers_the_earliest_tie() -> None:
    assert first_max(np.array([1.0, 3.0, 3.0 * (1 + 1e-12), math.nan])) == 1
    assert first_max(np.array([2.0, 1.0])) == 0


def test_first_max_without_a_finite_value_fails_clearly() -> None:
    with pytest.raises(ValueError, match="finite"):
        first_max(np.array([math.nan, math.nan]))


def test_third_difference_coefficients() -> None:
    steps = np.eye(5)
    assert np.array_equal(third_difference(steps), [[-1, 3, -3, 1, 0], [0, -1, 3, -3, 1]])
    t = np.arange(6, dtype=np.float64)
    quadratic = np.stack([2.0 - 0.5 * t + 3.0 * t**2, -4.9 * t**2], axis=1)
    assert np.allclose(third_difference(quadratic), 0.0, atol=1e-12)
    assert third_difference(t[:3]).shape == (0,)
    assert third_difference(np.zeros((3, 3))).shape == (0, 3)
