"""R4_jump rigid2_dfloor_shape: keep a verdict only for a peak shaped like a displaced body."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import test_r4_jump_null_tail as gate

from physverify.checks._base import CheckInputs
from physverify.checks.variants.r4_jump_step_shape import StepShapeJumpCheck, StepShapeJumpParams
from physverify.physics.measures.jump_shape import (
    SHAPES,
    Shape,
    fit_shapes,
    response,
    row_correlation,
)
from physverify.physics.measures.track_jump import noise_jumps, whitened_jumps
from physverify.schema.checks import AbstainReason, CheckResult, Verdict

T = np.arange(30)


def shaped(result: CheckInputs) -> CheckResult:
    (found,) = StepShapeJumpCheck(StepShapeJumpParams()).evaluate(result)
    return found


def disturbed(shape: Shape, size: float, at: int = 14, seed: int = 0) -> np.ndarray:
    smooth = np.c_[0.5 * T, 0.02 * T**2, np.zeros(T.size)]
    direction = np.array([0.6, -0.48, 0.64])
    bump = {
        "step": (at < T).astype(float),
        "kink": np.maximum(T - at, 0).astype(float),
        "outlier": (at == T).astype(float),
    }[shape]
    noise = np.random.default_rng(seed).normal(size=smooth.shape)
    return smooth + bump[:, None] * size * direction + noise


def test_row_correlation_is_that_of_overlapping_stencils() -> None:
    assert [row_correlation(lag) for lag in range(5)] == [1.0, -0.75, 0.3, -0.05, 0.0]


def test_responses_have_the_textbook_patterns() -> None:
    rows = np.arange(5.0, 12.0)
    scale = np.sqrt(20.0)
    assert np.allclose(response("step", 8, rows) * scale, [0, 0, 1, -2, 1, 0, 0])
    assert np.allclose(response("kink", 8, rows) * scale, [0, 0, 1, -1, 0, 0, 0])
    assert np.allclose(response("outlier", 8, rows) * scale, [0, 1, -3, 3, -1, 0, 0])


@pytest.mark.parametrize("shape", SHAPES)
def test_each_shape_is_told_apart_at_a_clear_peak(shape: Shape) -> None:
    whitened = whitened_jumps(disturbed(shape, 40.0), np.ones(3))
    row = int(np.nanargmax(np.linalg.norm(whitened, axis=1)))
    assert fit_shapes(whitened, row).best == shape


def test_whitened_jumps_have_the_norm_of_noise_jumps() -> None:
    positions, sigma = disturbed("step", 5.0), np.array([1.0, 2.0, 0.5])
    norms = np.linalg.norm(whitened_jumps(positions, sigma), axis=1)
    assert np.array_equal(norms, noise_jumps(positions, sigma), equal_nan=True)


def test_a_displaced_body_keeps_the_gated_verdict_input() -> None:
    inputs = gate.shifted(gate.noisy_rest(), [60], 40 * gate.SIGMA)
    found, base = shaped(inputs), gate.gated(inputs)
    assert found.verdict is Verdict.UNSCORED and found.details["shape"] == "step"
    assert found.magnitude == base.magnitude


def test_a_velocity_kink_abstains() -> None:
    inputs = gate.noisy_rest()
    position = inputs.states.position.copy()
    position[60:, 0, 0] += np.arange(position.shape[0] - 60) * 40 * gate.SIGMA
    found = shaped(replace(inputs, states=replace(inputs.states, position=position)))
    assert found.verdict is Verdict.ABSTAIN
    assert found.abstain_reason is AbstainReason.UNKNOWN_UNCERTAINTY
    assert found.details["shape"] == "kink"


def test_a_window_that_cannot_tell_the_shapes_apart_is_not_a_step() -> None:
    whitened = whitened_jumps(disturbed("step", 40.0), np.ones(3))
    row = int(np.nanargmax(np.linalg.norm(whitened, axis=1)))
    whitened[np.arange(whitened.shape[0]) != row] = np.nan
    fit = fit_shapes(whitened, row)
    assert fit.rows == 1 and fit.best != "step"


def test_a_peak_below_the_shape_level_keeps_the_gated_result() -> None:
    inputs = gate.noisy_rest()
    found, base = shaped(inputs), gate.gated(inputs)
    assert "shape" not in found.details
    assert found.verdict is base.verdict and found.magnitude == base.magnitude


def test_a_window_too_short_for_a_step_is_not_a_step() -> None:
    whitened = whitened_jumps(disturbed("step", 40.0), np.ones(3))
    row = int(np.nanargmax(np.linalg.norm(whitened, axis=1)))
    whitened[np.abs(np.arange(whitened.shape[0]) - row) > 1] = np.nan
    whitened[row + 1] = np.nan
    fit = fit_shapes(whitened, row)
    assert fit.rows == 2 and fit.best != "step"
