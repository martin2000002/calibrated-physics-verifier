"""Timing noise of a track: sigma_delta fitted on the moving cells of its third differences."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.physics.finite_diff import third_difference
from physverify.physics.measures.track_jump import jump_over_noise, noise_jumps
from physverify.physics.timing_noise import (
    TimingFit,
    moving_cells,
    stencil_sigma,
    stencil_speed,
    timing_sigma,
)
from physverify.physics.tracks import MIN_NOISE_STENCILS
from physverify.physics.uncertainty import robust_scale

SIGMA = 0.01
VELOCITY = np.array([0.3, 0.1, 0.2])
"""Per row, well above sqrt(20) SIGMA on every axis."""


def uniform(frames: int, delta: float, seed: int, sigma: float = SIGMA) -> np.ndarray:
    """A body at VELOCITY whose row r is taken at r + delta_r, delta ~ N(0, delta^2), plus white
    position noise ``sigma``."""
    rng = np.random.default_rng(seed)
    instants = np.arange(frames) + rng.normal(0.0, delta, frames)
    return np.asarray(np.outer(instants, VELOCITY) + rng.normal(0.0, sigma, (frames, 3)))


def fit(positions: np.ndarray, sigma_hat: np.ndarray | None = None) -> TimingFit:
    d3 = third_difference(positions)
    sigma = fitted_sigma(positions) if sigma_hat is None else sigma_hat
    usable = np.all(np.isfinite(d3), axis=-1)
    return timing_sigma(d3, sigma, stencil_speed(positions), usable, MIN_NOISE_STENCILS)


def fitted_sigma(positions: np.ndarray) -> np.ndarray:
    """sigma-hat as the track fits it: robust_scale(d3) / sqrt(20) per axis."""
    d3 = third_difference(positions)
    return np.array([robust_scale(d3[:, axis]) for axis in range(3)]) / math.sqrt(20.0)


def test_exact_positions_and_a_still_track_have_no_timing_noise() -> None:
    still = np.tile([0.3, 0.2, 4.0], (40, 1))
    assert fit(still, np.full(3, SIGMA)) == TimingFit(0.0, 0)
    assert fit(uniform(40, 0.1, seed=1), np.zeros(3)) == TimingFit(0.0, 0)
    noisy = still + np.random.default_rng(2).normal(0.0, SIGMA, still.shape)
    assert fit(noisy).sigma_frames == 0.0


def test_too_few_moving_cells_give_no_timing_noise() -> None:
    positions = uniform(40, 0.1, seed=3)
    positions[:, 1:] = positions[0, 1:]
    positions[9:, 0] = positions[8, 0]
    result = fit(positions, np.full(3, SIGMA))
    assert 0 < result.cells < MIN_NOISE_STENCILS
    assert result.sigma_frames == 0.0


def test_a_track_no_louder_in_motion_than_its_sigma_has_no_timing_noise() -> None:
    positions = uniform(200, 0.0, seed=4)
    cells = moving_cells(
        third_difference(positions),
        np.full(3, 2 * SIGMA),
        stencil_speed(positions),
        np.ones(197, dtype=bool),
    )
    assert cells.excess(0.0) <= 0.0
    assert fit(positions, np.full(3, 2 * SIGMA)) == TimingFit(0.0, cells.count)


def test_the_timing_noise_of_a_uniform_track_given_the_true_position_sigma() -> None:
    result = fit(uniform(400, 0.1, seed=0), np.full(3, SIGMA))
    assert result.sigma_frames == pytest.approx(0.1, rel=0.2)
    assert result.cells == 3 * 397


def test_position_noise_alone_gives_almost_no_timing_noise() -> None:
    assert fit(uniform(400, 0.0, seed=6), np.full(3, SIGMA)).sigma_frames <= 0.02


def test_a_step_on_a_moving_track_barely_moves_sigma_delta_and_is_still_seen() -> None:
    positions = uniform(400, 0.1, seed=7)
    stepped = positions.copy()
    stepped[200:, 1] += 50 * SIGMA
    sigma = np.full(3, SIGMA)
    clean, moved = fit(positions, sigma), fit(stepped, sigma)
    assert moved.sigma_frames == pytest.approx(clean.sigma_frames, rel=0.1)
    timed = stencil_sigma(sigma, stencil_speed(stepped), moved.sigma_frames)
    values = noise_jumps(stepped, timed)
    assert int(np.nanargmax(values)) in (199, 200)
    assert jump_over_noise(stepped, timed) > 10.0


def test_the_speed_of_a_stencil_with_a_step_is_the_median_of_its_other_steps() -> None:
    positions = np.outer(np.arange(12.0), [0.1, 0.0, -0.2])
    positions[6:, 0] += 0.5
    speed = stencil_speed(positions)
    assert speed.shape == (9, 3)
    assert np.allclose(speed[3:6], [0.1, 0.0, 0.2])


def test_the_speed_needs_four_rows() -> None:
    assert stencil_speed(np.zeros((3, 3))).shape == (0, 3)


def test_g_does_not_increase_with_s() -> None:
    positions = uniform(120, 0.2, seed=8)
    cells = moving_cells(
        third_difference(positions),
        np.full(3, SIGMA),
        stencil_speed(positions),
        np.ones(117, dtype=bool),
    )
    grid = np.concatenate([[0.0], np.geomspace(1e-4, 1e3, 200)])
    values = np.array([cells.excess(float(s)) for s in grid])
    assert np.all(np.diff(values) <= 0.0)
    assert values[-1] == pytest.approx(-1.0, abs=1e-3)


def test_the_stencil_sigma_grows_with_the_speed_and_keeps_sigma_hat_where_unknown() -> None:
    speed = np.array([[0.0, 0.1, math.nan], [1.0, 0.2, 0.3]])
    sigma = stencil_sigma([0.01, 0.02, 0.03], speed, 0.1)
    assert sigma[0].tolist() == pytest.approx([0.01, math.hypot(0.02, 0.01), 0.03])
    assert sigma[1].tolist() == pytest.approx(
        [math.hypot(0.01, 0.1), math.hypot(0.02, 0.02), math.hypot(0.03, 0.03)]
    )


def fall_and_rest(seed: int, delta: float) -> np.ndarray:
    """A body that slides and falls for 150 rows and rests for 450, its rows taken at r + delta_r,
    delta ~ N(0, delta^2), with white position noise SIGMA."""
    rng = np.random.default_rng(seed)
    instants = np.minimum(np.arange(600) + rng.normal(0.0, delta, 600), 150.0)
    path = np.column_stack(
        [0.3 * instants, -0.2 * instants, 4.0 - 0.001 * instants**2 - 0.1 * instants]
    )
    return np.asarray(path + rng.normal(0.0, SIGMA, path.shape))


def test_after_a_fall_and_rest_sigma_delta_is_recovered_and_the_jumps_shrink() -> None:
    positions = fall_and_rest(seed=9, delta=0.1)
    sigma = fitted_sigma(positions)
    result = fit(positions, sigma)
    assert result.sigma_frames == pytest.approx(0.1, rel=0.2)
    timed = stencil_sigma(sigma, stencil_speed(positions), result.sigma_frames)
    assert jump_over_noise(positions, timed) < jump_over_noise(positions, sigma)


def test_on_a_uniform_track_sigma_hat_absorbs_the_timing_noise() -> None:
    positions = uniform(240, 0.1, seed=10)
    sigma = fitted_sigma(positions)
    result = fit(positions, sigma)
    assert result.sigma_frames <= 0.02
    timed = stencil_sigma(sigma, stencil_speed(positions), result.sigma_frames)
    assert jump_over_noise(positions, timed) == pytest.approx(
        jump_over_noise(positions, sigma), rel=0.01
    )
