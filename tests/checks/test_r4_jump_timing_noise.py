"""R4_jump rigid2_dfloor_timing: rigid2_dfloor with the stencil noise grown by sigma_delta v."""

from __future__ import annotations

from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks._base import CheckInputs, validate_check
from physverify.checks.r4_track_jump import NoiseContinuityCheck, NoiseJumpParams, TrackNoise
from physverify.checks.variants._base import validate_variant
from physverify.checks.variants._declared_depth_jump import DeclaredDepthJumpCheck
from physverify.checks.variants._rigid_jump import RigidJumpParams
from physverify.checks.variants.r4_jump_timing_noise import TimingJumpCheck
from physverify.physics.finite_diff import third_difference, third_difference_sigma
from physverify.physics.tracks import noise_stencils
from physverify.schema.checks import CheckResult

SIGMA = 0.01
TIMING_KEYS = {"timing_sigma_frames", "timing_moving_cells", "timing_sigma_eff"}


def dfloor(inputs: CheckInputs) -> CheckResult:
    (result,) = DeclaredDepthJumpCheck(RigidJumpParams()).evaluate(inputs)
    return result


def timed(inputs: CheckInputs) -> CheckResult:
    (result,) = TimingJumpCheck(RigidJumpParams()).evaluate(inputs)
    return result


def with_track(positions: np.ndarray) -> CheckInputs:
    """The rolling ball's inputs with ``positions`` as its track and SIGMA declared per axis."""
    inputs = sc.rolling_ball(speed=0.0, frames=positions.shape[0]).inputs()
    states = replace(
        inputs.states,
        position=positions[:, None, :].copy(),
        position_sigma=np.full(inputs.states.position_sigma.shape, SIGMA),
    )
    return replace(inputs, states=states)


def slide_and_rest(seed: int, delta: float, frames: int = 240, moving: float = 60.0) -> np.ndarray:
    """A body that slides for ``moving`` rows and rests, its rows taken at r + delta_r with
    delta ~ N(0, delta^2), plus white position noise SIGMA."""
    rng = np.random.default_rng(seed)
    instants = np.minimum(np.arange(frames) + rng.normal(0.0, delta, frames), moving)
    path = np.outer(instants, [0.3, -0.2, 0.25]) + np.array([0.0, 0.0, 4.0])
    return np.asarray(path + rng.normal(0.0, SIGMA, path.shape))


def test_the_variant_declares_its_version_and_passes_the_registry_hooks() -> None:
    assert vars(TimingJumpCheck)["version"] == "1"
    assert vars(TimingJumpCheck)["variant"] == "rigid2_dfloor_timing"
    assert TimingJumpCheck.check_id == "R4_jump"
    assert TimingJumpCheck.magnitude_unit == "jump_over_noise"
    assert "replaced_unit" not in vars(TimingJumpCheck)
    validate_check(TimingJumpCheck)
    validate_variant(TimingJumpCheck)


def stencils_of(inputs: CheckInputs) -> tuple[np.ndarray, np.ndarray]:
    return noise_stencils(inputs.states, 0, np.zeros(inputs.states.n_frames, dtype=bool))


def test_the_base_seams_return_sigma_hat_itself_and_no_details() -> None:
    inputs = sc.teleport().inputs()
    noise = TrackNoise(np.array([0.01, 0.02, 0.03]), "track", 0)
    check = NoiseContinuityCheck(NoiseJumpParams())
    sigma, details = check._stencil_sigma(inputs, 0, stencils_of(inputs), noise)
    assert sigma is noise.sigma
    assert details == {}
    assert check._peak_details(4, np.arange(30.0).reshape(10, 3)) == {}


def test_the_peak_noise_is_the_stencil_of_the_peak_row() -> None:
    check = TimingJumpCheck(RigidJumpParams())
    per_stencil = np.arange(30.0).reshape(10, 3)
    assert check._peak_details(4, per_stencil) == {"timing_sigma_eff": [9.0, 10.0, 11.0]}
    assert check._peak_details(4, np.array([0.1, 0.2, 0.3])) == {
        "timing_sigma_eff": [0.1, 0.2, 0.3]
    }


def test_exact_positions_keep_sigma_hat_and_no_timing_noise() -> None:
    inputs = sc.teleport().inputs()
    noise = TrackNoise(np.array([0.001, 0.001, 0.001]), "exact", 3)
    check = TimingJumpCheck(RigidJumpParams())
    sigma, details = check._stencil_sigma(inputs, 0, stencils_of(inputs), noise)
    assert sigma is noise.sigma
    assert details["timing_sigma_frames"] == 0.0
    assert details["timing_moving_cells"] == 0
    result, base = timed(inputs), dfloor(inputs)
    assert result.details["noise_source"] == "exact"
    assert result.magnitude == base.magnitude and result.sigma == base.sigma == 0.0


@pytest.mark.parametrize("seed", range(4))
def test_a_teleport_of_a_still_body_keeps_the_magnitude_of_rigid2_dfloor(seed: int) -> None:
    rng = np.random.default_rng(seed)
    positions = np.tile([0.3, 0.1, 4.0], (24, 1)) + rng.normal(0.0, SIGMA, (24, 3))
    positions[12:, 0] += 50 * SIGMA
    result, base = timed(with_track(positions)), dfloor(with_track(positions))
    assert result.details["noise_source"] == "track"
    assert result.details["timing_sigma_frames"] == 0.0
    assert result.magnitude == base.magnitude
    assert result.details["row"] == base.details["row"]


def test_timing_noise_after_a_slide_lowers_the_magnitude() -> None:
    inputs = with_track(slide_and_rest(seed=4, delta=0.1))
    result, base = timed(inputs), dfloor(inputs)
    assert result.details["timing_sigma_frames"] == pytest.approx(0.1, rel=0.2)
    assert result.magnitude < 0.8 * base.magnitude


def test_without_timing_noise_the_slide_keeps_the_magnitude() -> None:
    inputs = with_track(slide_and_rest(seed=2, delta=0.0))
    result, base = timed(inputs), dfloor(inputs)
    assert result.details["timing_sigma_frames"] == 0.0
    assert result.magnitude == base.magnitude


def test_the_timing_details_are_added_and_the_base_has_none() -> None:
    inputs = with_track(slide_and_rest(seed=4, delta=0.1))
    result, base = timed(inputs), dfloor(inputs)
    assert set(result.details) >= TIMING_KEYS
    assert not TIMING_KEYS & set(base.details)
    assert result.details["timing_moving_cells"] > 0
    peak = result.details["timing_sigma_eff"]
    assert isinstance(peak, list) and len(peak) == 3
    assert {k: v for k, v in result.details.items() if k not in TIMING_KEYS}.keys() == (
        base.details.keys()
    )


def test_the_peak_noise_restates_the_raw_statistic_at_the_peak_row() -> None:
    positions = slide_and_rest(seed=4, delta=0.1)
    result = timed(with_track(positions))
    row, peak = result.details["row"], result.details["timing_sigma_eff"]
    assert isinstance(row, int) and isinstance(peak, list)
    d3 = third_difference(positions)[row - 1]
    scale = third_difference_sigma(1.0, 1.0) * np.asarray(peak)
    assert float(np.linalg.norm(d3 / scale)) == pytest.approx(result.details["raw_jump_over_noise"])
