import math
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks._base import Check, CheckInputs
from physverify.checks._common import CommonParams, first_max
from physverify.checks.r4_integrity import (
    PLUGINS,
    ContinuityCheck,
    PermanenceCheck,
    PermanenceParams,
    SizeConstancyCheck,
    SizeParams,
)
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.constants import STANDARD_GRAVITY

BALL_LENGTH = 2 * sc.BALL_RADIUS


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


def run_one(check: Check, case: sc.Case) -> CheckResult:
    results = check.evaluate(inputs(case))
    assert len(results) == 1
    result = results[0]
    if result.frame_span is not None:
        native = set(case.states.frame_index.tolist())
        assert set(result.frame_span) <= native
    return result


def missing(case: sc.Case) -> CheckResult:
    return run_one(PermanenceCheck(PermanenceParams()), case)


def size(case: sc.Case) -> CheckResult:
    return run_one(SizeConstancyCheck(SizeParams()), case)


def jump(case: sc.Case) -> CheckResult:
    return run_one(ContinuityCheck(CommonParams()), case)


def test_plugins_satisfy_the_check_protocol() -> None:
    ids = {cls.check_id for cls in PLUGINS}
    assert ids == {"R4_missing", "R4_size", "R4_jump"}
    for cls in PLUGINS:
        assert isinstance(cls(cls.Params()), Check)


# --- R4_missing -------------------------------------------------------------------------------


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


def test_missing_sigma_is_unknown_for_perceived_positions() -> None:
    result = missing(sc.with_sigma(sc.disappearance(4, 7), position=0.01))
    assert math.isnan(result.sigma)


def test_strided_disappearance_counts_native_frames() -> None:
    result = missing(sc.strided(sc.disappearance(4, 7), 2, quantize=True))
    assert result.magnitude == 4
    assert result.details["missing_rows"] == 2
    assert result.details["row_step"] == 2
    assert result.frame_span == (4, 6)


# --- R4_size ----------------------------------------------------------------------------------


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


# --- R4_jump ----------------------------------------------------------------------------------


@pytest.mark.parametrize("pitch", [0.0, 25.0])
def test_teleport_at_rest_is_detected(pitch: float) -> None:
    result = jump(sc.teleport(0.3, frame=16, pitch_deg=pitch))
    assert result.magnitude == pytest.approx(0.3 / BALL_LENGTH, abs=1e-9)
    assert result.frame_span == (14, 16)
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
    assert result.frame_span == (int(fi[k - 1]), int(fi[k + 1]))


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("offset", [0.0, 0.4])
@pytest.mark.parametrize("quantize", [False, True])
def test_bounce_impact_is_excluded(step: int, offset: float, quantize: bool) -> None:
    """Quantized contacts (a sub-frame impact snapped to one frame) still exclude the stencil."""
    case = sc.strided(sc.bounce(impact_frame=sc.EVENT_FRAME + offset), step, quantize=quantize)
    result = jump(case)
    smooth = STANDARD_GRAVITY * (step * sc.DT) ** 2 / BALL_LENGTH
    assert result.details["excluded_rows"] > 0
    assert result.magnitude is not None
    assert result.magnitude <= 1.01 * smooth


def test_jump_without_three_known_rows_abstains() -> None:
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
    expected = math.sqrt(6) * float(np.linalg.norm(along * sigma[0, 0])) / BALL_LENGTH
    assert result.sigma == pytest.approx(expected, rel=1e-9)


def test_jump_requires_contacts() -> None:
    case = replace(sc.rolling_ball(), contacts=None)
    with pytest.raises(ValueError, match="requires contacts"):
        ContinuityCheck(CommonParams()).evaluate(inputs(case))


def test_first_max_prefers_the_earliest_tie() -> None:
    assert first_max(np.array([1.0, 3.0, 3.0 * (1 + 1e-12), math.nan])) == 1
    assert first_max(np.array([2.0, 1.0])) == 0
