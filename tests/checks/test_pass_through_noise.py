"""Noise of R1_dyn v2: rate variances, the impulse statistic, the flip probability and the
propagated sigma of s = n (1 - p) against seeded Gaussian draws."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.checks.separate._pass_through_noise import (
    ROW_COVERAGE,
    band_z,
    count_sigma,
    descriptor_variance,
    evidence,
    evidence_sigma,
    flip_probability,
    impulse_statistic,
    magnitude_sigma,
    slope_variance,
)

FPS = 24.0
ROWS = 30
START = 12
FOCAL = 500.0
DRAWS = 2000
TOLERANCE = 0.10
FRAMES_PER_ROW = 2
# Gap minus band of each row of a pass-through run and its neighbours, in row sigmas.
MARGINS = np.array([-0.5, -3.0, -2.0, -0.5, -1.0, -2.5, -2.0, -1.5, -0.2, 0.4])


def times() -> np.ndarray:
    return np.arange(ROWS) / FPS


def rate(t: np.ndarray, values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    centred = t[rows] - t[rows].mean()
    out: np.ndarray = centred @ (values[rows] - values[rows].mean(axis=0)) / (centred @ centred)
    return out


def change_variance(row: float, size_rel: float, descriptor: np.ndarray) -> np.ndarray:
    size = np.full(ROWS, 80.0)
    variance = descriptor_variance(
        descriptor, size, FOCAL, np.full(ROWS, row), np.full(ROWS, size_rel)
    )
    t = times()
    before, after = np.arange(START), np.arange(START, ROWS)
    return slope_variance(t, variance, before, 1.0) + slope_variance(t, variance, after, 1.0)


def test_exact_inputs_have_zero_sigma() -> None:
    descriptor = np.outer(times(), [0.5, 0.1, -0.3])
    assert not change_variance(0.0, 0.0, descriptor).any()
    flip = flip_probability(MARGINS, np.zeros(MARGINS.size), None)
    assert count_sigma(flip, FRAMES_PER_ROW) == 0.0
    assert magnitude_sigma(14, 0.2, 0.0, 0.0) == 0.0


@pytest.mark.parametrize("factor", [0.5, 2.0, 7.0])
def test_the_rate_change_sigma_is_linear_in_the_row_sigma(factor: float) -> None:
    descriptor = np.outer(times(), [0.5, 0.1, -0.3])
    base = np.sqrt(change_variance(1.5, 0.0, descriptor))
    scaled = np.sqrt(change_variance(1.5 * factor, 0.0, descriptor))
    assert scaled[:2] == pytest.approx(factor * base[:2], rel=1e-12)
    assert not scaled[2]


def test_the_slope_variance_scales_with_the_autocorrelation() -> None:
    t, variance, rows = times(), np.ones((ROWS, 3)), np.arange(START)
    assert slope_variance(t, variance, rows, 2.5) == pytest.approx(
        2.5 * slope_variance(t, variance, rows, 1.0)
    )


def test_without_an_impulse_the_evidence_is_a_maximum_of_uniforms() -> None:
    assert evidence(0.0) == 0.0
    assert evidence_sigma([0.0]) == pytest.approx(1.0 / math.sqrt(12.0), rel=1e-4)
    assert evidence_sigma([1.0, 2.5]) == pytest.approx(math.sqrt(1.0 / 18.0), rel=1e-4)
    assert evidence_sigma([400.0]) < 1e-3


def test_the_tail_law_flips_as_often_as_the_normal_at_the_band_edge() -> None:
    edge, sigma = np.array([band_z()]), np.ones(1)
    normal = flip_probability(edge, sigma, None)
    assert normal == pytest.approx((1.0 - ROW_COVERAGE) / 2.0)
    assert flip_probability(edge, sigma, 3.0) == pytest.approx(normal)
    assert flip_probability(4.0 * edge, sigma, 3.0) > flip_probability(4.0 * edge, sigma, None)


def test_the_sigma_of_s_matches_seeded_gaussian_draws() -> None:
    """A pass-through (no impulse) with row and descriptor noise: the empirical spread of s
    over ``DRAWS`` draws is within ``TOLERANCE`` of the root mean square sigma the check
    reports on those same draws."""
    rng = np.random.default_rng(20261106)
    t, noise = times(), 0.01
    before, after = np.arange(START), np.arange(START, ROWS)
    bodies = [np.outer(t, [0.5, 0.1, -0.3]), np.zeros((ROWS, 3))]
    variance = noise**2 * np.ones((ROWS, 3))
    rate_variance = slope_variance(t, variance, before, 1.0) + slope_variance(
        t, variance, after, 1.0
    )
    row_sigma = np.ones(MARGINS.size)
    s, reported = np.empty(DRAWS), np.empty(DRAWS)
    for draw in range(DRAWS):
        statistics = []
        for body in bodies:
            seen = body + rng.normal(0.0, noise, body.shape)
            change = rate(t, seen, after) - rate(t, seen, before)
            statistics.append(impulse_statistic(change, rate_variance))
        impulse = max(evidence(q) for q in statistics)
        margins = MARGINS + rng.normal(0.0, 1.0, MARGINS.size)
        frames = FRAMES_PER_ROW * int(np.sum(margins <= 0.0))
        frames_sigma = count_sigma(flip_probability(margins, row_sigma, None), FRAMES_PER_ROW)
        s[draw] = frames * (1.0 - impulse)
        reported[draw] = magnitude_sigma(frames, impulse, frames_sigma, evidence_sigma(statistics))
    expected = math.sqrt(float(np.mean(reported**2)))
    assert abs(float(np.std(s)) - expected) <= TOLERANCE * expected
