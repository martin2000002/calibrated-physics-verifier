"""it4 change A: R4_jump v2, the jump in units of the object's own noise (plan-it4 T1-T7, T9;
judgement J2, J6, J8, J9). Analytic scenes seen by a pinhole of ``sc.FOCAL_PX`` pixels."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks._base import Check, CheckInputs
from physverify.checks.r4_jump_noise import NoiseContinuityCheck, NoiseJumpParams
from physverify.cli.context import PLUGIN_KINDS
from physverify.physics.finite_diff import third_difference
from physverify.physics.measures.r4_jump_noise import jump_over_noise, noise_jumps
from physverify.physics.tracks import QUANTIZATION, impact_stencil_mask
from physverify.schema.checks import AbstainReason, CheckResult, Verdict
from physverify.verification.inputs import ClipInputs
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import VerificationRunner

DEPTH = 4.0
"""Camera depth of the rolling ball and the teleport scenes (sc.CAMERA_ORIGIN at y = -4)."""
FLOOR = DEPTH / (sc.FOCAL_PX * QUANTIZATION)


def check(**params: Any) -> NoiseContinuityCheck:
    return NoiseContinuityCheck(NoiseJumpParams(**params))


def one(inputs: CheckInputs, **params: Any) -> CheckResult:
    (result,) = check(**params).evaluate(inputs)
    return result


def with_positions(inputs: CheckInputs, position: np.ndarray, sigma: float) -> CheckInputs:
    states = replace(
        inputs.states,
        position=position,
        position_sigma=np.full(inputs.states.position_sigma.shape, sigma),
    )
    return replace(inputs, states=states)


def noisy_rest(sigma: float, seed: int, frames: int = 24) -> CheckInputs:
    """A ball at rest at depth DEPTH with white position noise ``sigma`` (declared)."""
    case = sc.rolling_ball(speed=0.0, frames=frames)
    noise = np.random.default_rng(seed).normal(0.0, sigma, case.states.position.shape)
    return with_positions(case.inputs(), case.states.position + noise, sigma)


# --- T1: white noise, invariances ---------------------------------------------------------------


def test_t1_white_noise_peaks_like_the_maximum_of_chi3_norms() -> None:
    """Median of m over 2000 noisy tracks (sigma-hat measured on each) against the median of the
    same maximum with the true sigma, within 5 %."""
    sigma, draws = 0.02, 2000
    base = noisy_rest(sigma, 0)
    rng = np.random.default_rng(1)
    measured, truth = [], []
    rest = base.states.position.copy()
    rest[:] = rest[0]
    for _ in range(draws):
        noise = rng.normal(0.0, sigma, rest.shape)
        measured.append(one(with_positions(base, rest + noise, sigma)).magnitude)
        truth.append(jump_over_noise(noise[:, 0], np.full(3, sigma)))
    ratio = float(np.median(measured)) / float(np.median(truth))
    assert 0.95 <= ratio <= 1.05


def scaled(inputs: CheckInputs, factor: float) -> CheckInputs:
    states = inputs.states
    return replace(
        inputs,
        states=replace(
            states,
            position=states.position * factor,
            position_sigma=states.position_sigma * factor,
            size=states.size * factor,
            size_sigma=states.size_sigma * factor,
        ),
    )


def with_gravity(inputs: CheckInputs, g: np.ndarray) -> CheckInputs:
    states = inputs.states
    drift = 0.5 * states.t[:, None, None] ** 2 * g[None, None, :]
    return replace(inputs, states=replace(states, position=states.position + drift))


@pytest.mark.parametrize("factor", [0.01, 100.0])
def test_t1_the_value_does_not_depend_on_the_scale(factor: float) -> None:
    noisy = with_positions(sc.teleport().inputs(), noisy_rest(0.01, 3, 32).states.position, 0.01)
    noisy = replace(noisy, states=replace(noisy.states, position=noisy.states.position))
    base = one(noisy)
    moved = one(scaled(noisy, factor))
    assert moved.magnitude == pytest.approx(base.magnitude, rel=1e-9)
    assert moved.details["row"] == base.details["row"]


@pytest.mark.parametrize("direction", [(0.0, 9.81, 0.0), (3.0, -2.0, 7.0)])
def test_t1_the_value_does_not_depend_on_gravity(direction: tuple[float, float, float]) -> None:
    inputs = with_positions(
        sc.teleport().inputs(),
        sc.teleport().states.position + np.random.default_rng(4).normal(0.0, 0.01, (32, 1, 3)),
        0.01,
    )
    base = one(inputs)
    pulled = one(with_gravity(inputs, np.array(direction)))
    assert pulled.magnitude == pytest.approx(base.magnitude, rel=1e-6)


# --- T2, T3: a step -------------------------------------------------------------------------


def stepped(inputs: CheckInputs, row: int, offset: np.ndarray) -> CheckInputs:
    position = inputs.states.position.copy()
    position[row + 1 :, 0] += offset
    return replace(inputs, states=replace(inputs.states, position=position))


def test_t2_a_step_is_twice_its_size_over_the_noise_of_its_stencil() -> None:
    sigma, row = 0.01, 11
    step = 100 * sigma
    result = one(stepped(noisy_rest(sigma, 5), row, np.array([step, 0.0, 0.0])))
    assert result.details["row"] == row
    assert result.details["noise_source"] == "track"
    sigma_hat = np.array(result.details["sigma_over_length"]) * 2 * sc.BALL_RADIUS
    assert 0.6 < sigma_hat[0] / sigma < 1.6
    expected = 2 * step / (math.sqrt(20) * sigma_hat[0])
    assert abs(result.magnitude - expected) <= 4.0


def test_t3_exact_positions_use_the_pixel_floor_on_every_axis() -> None:
    case = sc.rolling_ball()
    step = 5 * DEPTH / sc.FOCAL_PX
    result = one(stepped(case.inputs(), 15, np.array([step, 0.0, 0.0])))
    assert result.details["noise_source"] == "exact"
    assert result.details["floored_axes"] == 3
    assert result.magnitude == pytest.approx(10 * math.sqrt(12) / math.sqrt(20), rel=1e-9)
    assert result.sigma == 0.0
    sigma_hat = np.array(result.details["sigma_over_length"]) * 2 * sc.BALL_RADIUS
    assert sigma_hat == pytest.approx(np.full(3, FLOOR), rel=1e-12)


# --- T4, T4b: too few stencils (judgement J2) -------------------------------------------------


def short_track(sigma: float) -> CheckInputs:
    """10 rows: 7 stencils, below MIN_NOISE_STENCILS."""
    return noisy_rest(sigma, 6, frames=10)


def test_t4_too_few_stencils_abstain_without_the_declared_branch() -> None:
    result = one(short_track(0.01), few_stencils="abstain")
    assert result.abstain_reason is AbstainReason.TOO_FEW_FRAMES
    assert result.details["usable_stencils"] == 7


def test_t4b_too_few_stencils_take_the_declared_sigma() -> None:
    inputs = short_track(0.01)
    result = one(inputs)
    assert result.verdict is Verdict.UNSCORED
    assert result.details["noise_source"] == "declared"
    sigma_hat = np.array(result.details["sigma_over_length"]) * 2 * sc.BALL_RADIUS
    assert sigma_hat == pytest.approx(np.full(3, 0.01), rel=1e-12)
    expected = jump_over_noise(inputs.states.position[:, 0], np.full(3, 0.01))
    assert result.magnitude == pytest.approx(expected, rel=1e-12)
    assert result.sigma == 1.0


def test_t4b_the_declared_sigma_is_floored() -> None:
    inputs = short_track(FLOOR / 10)
    result = one(inputs)
    depth = float(np.median(inputs.states.position[:, 0, 2]))
    sigma_hat = np.array(result.details["sigma_over_length"]) * 2 * sc.BALL_RADIUS
    assert sigma_hat == pytest.approx(np.full(3, depth / (sc.FOCAL_PX * QUANTIZATION)), rel=1e-9)
    assert result.details["floored_axes"] == 3


def test_t4b_without_any_sigma_too_few_stencils_abstain() -> None:
    inputs = short_track(0.01)
    unknown = np.full(inputs.states.position_sigma.shape, math.nan)
    inputs = replace(inputs, states=replace(inputs.states, position_sigma=unknown))
    assert one(inputs).abstain_reason is AbstainReason.TOO_FEW_FRAMES


# --- T5: impacts --------------------------------------------------------------------------------


@pytest.mark.parametrize("step", [1, 2])
def test_t5_the_stencils_of_a_marked_impact_do_not_count(step: int) -> None:
    case = sc.strided(sc.bounce(), step, quantize=True)
    result = one(case.inputs())
    assert case.contacts is not None
    mask = impact_stencil_mask("ball", case.contacts, case.states.frame_index)
    assert result.details["excluded_rows"] == int(mask.sum())
    assert result.magnitude == pytest.approx(0.0, abs=1e-9)
    unmarked = replace(case.contacts, records=())
    assert one(replace(case.inputs(), contacts=unmarked)).magnitude > 10.0


# --- T6: the camera (judgement J6 iv) ------------------------------------------------------------


def test_t6_without_a_camera_the_runner_abstains_by_requires() -> None:
    inputs = replace(sc.teleport().inputs(), camera=None)
    item = ClipInputs(inputs.clip, inputs, None)

    class One:
        def load(self, clips: Any) -> Any:
            return iter([item])

    checks: list[Check] = [check()]
    report = VerificationRunner(checks, RawPolicy(), {}).run(One(), [inputs.clip])
    (result,) = report.results
    assert result.abstain_reason is AbstainReason.MISSING_INPUT
    assert result.details["missing"] == ["camera"]


def test_t6_the_check_itself_refuses_a_missing_camera() -> None:
    with pytest.raises(ValueError, match="requires contacts and camera"):
        check().evaluate(replace(sc.teleport().inputs(), camera=None))


@pytest.mark.parametrize("broken", ["focal", "depth"])
def test_t6_no_pixel_floor_is_unknown_uncertainty(broken: str) -> None:
    inputs = sc.teleport().inputs()
    if broken == "focal":
        assert inputs.camera is not None
        inputs = replace(inputs, camera=sc.pinhole(inputs.clip, focal=0.0))
    else:
        flipped = inputs.states.position * np.array([1.0, 1.0, -1.0])
        inputs = replace(inputs, states=replace(inputs.states, position=flipped))
    assert one(inputs).abstain_reason is AbstainReason.UNKNOWN_UNCERTAINTY


# --- T7: sigma and details ----------------------------------------------------------------------


def test_t7_sigma_is_one_unless_the_positions_are_exact() -> None:
    exact = one(sc.teleport().inputs())
    assert exact.sigma == 0.0
    noisy = one(sc.with_sigma(sc.teleport(), position=0.001).inputs())
    assert noisy.sigma == 1.0


def test_t7_details_carry_the_estimated_physical_jump() -> None:
    inputs = sc.teleport(delta=0.3).inputs()
    result = one(inputs)
    assert result.unit == "jump_over_noise"
    assert result.details["jump_over_size_estimated"] == pytest.approx(1.5, rel=1e-9)
    floor = float(np.median(inputs.states.position[:, 0, 2])) / (sc.FOCAL_PX * QUANTIZATION)
    assert result.details["sigma_over_length"] == pytest.approx([floor / 0.2] * 3, rel=1e-9)
    assert result.frame_span is not None
    assert result.details["row"] == 15


def test_noise_jumps_are_nan_outside_the_stencils() -> None:
    p = np.random.default_rng(7).normal(size=(12, 3))
    values = noise_jumps(p, np.ones(3))
    assert np.isnan(values[[0, -2, -1]]).all()
    assert values[1:-2] == pytest.approx(
        np.linalg.norm(third_difference(p), axis=1) / math.sqrt(20)
    )
    excluded = np.zeros(12, dtype=bool)
    excluded[5] = True
    assert math.isnan(noise_jumps(p, np.ones(3), excluded)[5])
    assert math.isnan(jump_over_noise(p[:3], np.ones(3)))


def test_the_peak_value_has_unit_sigma_at_a_fixed_noise_scale() -> None:
    """J8: sigma = 1 is the per-row sigma at the peak: with sigma-hat fixed, the value at a large
    step's row varies with standard deviation 1 under the noise it is measured in."""
    rng = np.random.default_rng(8)
    base = np.zeros((24, 3))
    base[12:, 0] += 1.0
    values = [
        noise_jumps(base + rng.normal(0.0, 0.01, base.shape), np.full(3, 0.01))[11]
        for _ in range(4000)
    ]
    assert float(np.std(values, ddof=1)) == pytest.approx(1.0, abs=0.05)


# --- T9: registry ------------------------------------------------------------------------------


def test_t9_the_registry_holds_one_r4_jump_the_v2() -> None:
    registry = PLUGIN_KINDS["checks"]()
    cls = registry["R4_jump"]
    assert cls is NoiseContinuityCheck
    assert cls.requires == frozenset({"contacts", "camera"})
    assert cls.magnitude_unit == "jump_over_noise"
    assert cls.perturbation_unit == "jump_over_size"
