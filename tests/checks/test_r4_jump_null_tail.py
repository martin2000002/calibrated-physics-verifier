"""R4_jump rigid2_dfloor_null: abstain when the stencils away from the peak reject the null."""

from __future__ import annotations

from dataclasses import replace

import check_scenarios as sc
import numpy as np

from physverify.checks._base import CheckInputs
from physverify.checks.variants._declared_depth_jump import DeclaredDepthJumpCheck
from physverify.checks.variants._null_tail_jump import NullTailJumpCheck, NullTailJumpParams
from physverify.physics.measures.null_tail import off_peak, tail_test
from physverify.schema.checks import AbstainReason, CheckResult, Verdict

SIGMA = 0.01
FRAMES = 120


def gated(inputs: CheckInputs) -> CheckResult:
    (result,) = NullTailJumpCheck(NullTailJumpParams()).evaluate(inputs)
    return result


def dfloor(inputs: CheckInputs) -> CheckResult:
    (result,) = DeclaredDepthJumpCheck(NullTailJumpParams()).evaluate(inputs)
    return result


def noisy_rest(seed: int = 3) -> CheckInputs:
    inputs = sc.rolling_ball(speed=0.0, frames=FRAMES).inputs()
    states = inputs.states
    noise = np.random.default_rng(seed).normal(0.0, SIGMA, states.position.shape)
    sigma = np.full(states.position_sigma.shape, SIGMA)
    return replace(
        inputs, states=replace(states, position=states.position + noise, position_sigma=sigma)
    )


def shifted(inputs: CheckInputs, rows: list[int], size: float) -> CheckInputs:
    """Every row from each of ``rows`` on moved by ``size`` along x."""
    position = inputs.states.position.copy()
    for row in rows:
        position[row:, 0, 0] += size
    return replace(inputs, states=replace(inputs.states, position=position))


def test_off_peak_leaves_out_the_guard_and_unknown_stencils() -> None:
    values = np.array([np.nan, 1.0, 2.0, 9.0, 2.0, 1.0, 3.0, np.nan])
    assert off_peak(values, 3, 2).tolist() == [3.0]
    assert off_peak(values, 3, 0).tolist() == [1.0, 2.0, 2.0, 1.0, 3.0]


def test_tail_test_counts_against_the_chi3_tail() -> None:
    quiet = tail_test(np.ones(200), 4.0)
    assert (quiet.exceed, quiet.stencils, quiet.p_value) == (0, 200, 1.0)
    loud = tail_test(np.r_[np.ones(190), np.full(10, 6.0)], 4.0)
    assert loud.exceed == 10 and loud.p_value < 1e-10
    assert tail_test(np.array([]), 4.0).p_value == 1.0


def test_one_jump_keeps_the_dfloor_verdict_input() -> None:
    inputs = shifted(noisy_rest(), [60], 40 * SIGMA)
    found, base = gated(inputs), dfloor(inputs)
    assert found.verdict is Verdict.UNSCORED
    assert found.magnitude == base.magnitude and found.details["row"] == base.details["row"]
    assert found.details["null_p_value"] > 0.01


def test_white_noise_does_not_reject_the_null() -> None:
    found = gated(noisy_rest())
    assert found.verdict is Verdict.UNSCORED and found.details["null_p_value"] > 0.01


def test_sporadic_jumps_along_the_track_abstain() -> None:
    inputs = shifted(noisy_rest(), [15, 30, 45, 75, 90, 105], 40 * SIGMA)
    found = gated(inputs)
    assert dfloor(inputs).verdict is Verdict.UNSCORED
    assert found.verdict is Verdict.ABSTAIN
    assert found.abstain_reason is AbstainReason.UNKNOWN_UNCERTAINTY
    assert found.details["null_exceed"] >= 5
