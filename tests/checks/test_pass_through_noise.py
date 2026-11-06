"""Noise of R1_dyn v2: the full covariance of the rate change, the impulse statistic, the flip
probability and the propagated sigma of s = n (1 - p), against seeded draws of the noise model
(pixel noise on u and v, relative size noise shared by the three components of D, each
autocorrelated across rows)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from physverify.checks.separate._pass_through_noise import (
    ROW_COVERAGE,
    band_z,
    change_covariance,
    change_weights,
    count_sigma,
    evidence,
    evidence_sigma,
    flip_probability,
    impulse_statistic,
    lag_correlation,
    magnitude_sigma,
    positive_definite,
    slope_weights,
)

Lags = tuple[tuple[float, ...], tuple[float, ...]]

FPS = 24.0
ROWS = 30
START = 12
FOCAL = 500.0
LENGTH = 0.5
DRAWS = 2000
NULL_DRAWS = 4000
TOLERANCE = 0.10
# Null-uniformity tolerances, fixed before running: each tail within 0.02 of 0.10 (about four
# binomial sd at 4000 draws) and a KS p-value above 0.001.
TAIL_TOLERANCE = 0.02
KS_FLOOR = 0.001
COVARIANCE_TOLERANCE = 0.08
FRAMES_PER_ROW = 2
WHITE: Lags = ((), ())
CORRELATED: Lags = ((0.6, 0.3, 0.1), (0.5, 0.2))
# Gap minus band of each row of a pass-through run and its neighbours, in row sigmas.
MARGINS = np.array([-0.5, -3.0, -2.0, -0.5, -1.0, -2.5, -2.0, -1.5, -0.2, 0.4])


def times() -> np.ndarray:
    return np.arange(ROWS) / FPS


def windows() -> tuple[np.ndarray, np.ndarray]:
    return np.arange(START), np.arange(START, ROWS)


def body() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image column, row and size of a body at constant 3D velocity: D is exactly linear."""
    t = times()
    x, y, z = 0.6 + 1.0 * t, 0.4 + 0.0 * t, 4.0 - 1.5 * t
    return FOCAL * x / z, FOCAL * y / z, FOCAL * LENGTH / z


def descriptor(u: np.ndarray, v: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.column_stack([u / s, v / s, FOCAL / s])


def correlated(rng: np.random.Generator, lags: tuple[float, ...], draws: int) -> np.ndarray:
    """(draws, ROWS) unit-variance Gaussian noise with autocorrelation ``lags``."""
    root = np.linalg.cholesky(lag_correlation(np.arange(ROWS), lags))
    out: np.ndarray = rng.standard_normal((draws, ROWS)) @ root.T
    return out


def covariance(
    d: np.ndarray, size: np.ndarray, pixel: float, relative: float, lags: Lags = WHITE
) -> np.ndarray:
    rows, weights = change_weights(times(), *windows())
    noise = (size, np.full(ROWS, pixel), np.full(ROWS, relative))
    return change_covariance(rows, weights, d, noise, lags[0], lags[1])


def changes(d: np.ndarray) -> np.ndarray:
    """(draws, 3) rate change after - before of a stack of descriptors (draws, ROWS, 3)."""
    rows, weights = change_weights(times(), *windows())
    out: np.ndarray = np.einsum("r,nri->ni", weights, d[:, rows])
    return out


def seen(
    rng: np.random.Generator, pixel: float, relative: float, lags: Lags, draws: int
) -> np.ndarray:
    """(draws, ROWS, 3) descriptors of the constant-velocity body read with noise."""
    u, v, s = body()
    noisy_u = u + pixel * correlated(rng, lags[0], draws)
    noisy_v = v + pixel * correlated(rng, lags[0], draws)
    noisy_s = s * (1.0 + relative * correlated(rng, lags[1], draws))
    return np.stack([noisy_u / noisy_s, noisy_v / noisy_s, FOCAL / noisy_s], axis=-1)


def first_order(
    rng: np.random.Generator, pixel: float, relative: float, lags: Lags, draws: int
) -> np.ndarray:
    """(draws, ROWS, 3) descriptors with the first-order noise the covariance is exact for:
    D + (du / s, dv / s, 0) - D e."""
    u, v, s = body()
    d = descriptor(u, v, s)
    pixels = pixel * np.stack([correlated(rng, lags[0], draws) for _ in range(2)], axis=-1)
    shared = relative * correlated(rng, lags[1], draws)
    out: np.ndarray = d - d * shared[..., None]
    out[..., :2] += pixels / s[:, None]
    return out


def lag_value(lags: tuple[float, ...], lag: int) -> float:
    return 1.0 if lag == 0 else (lags[lag - 1] if lag <= len(lags) else 0.0)


def test_the_change_weights_are_the_two_least_squares_slopes() -> None:
    t, (before, after) = times(), windows()
    values = np.column_stack([np.sin(t), t**2, 3.0 - t])
    rows, weights = change_weights(t, before, after)
    slope = [np.polyfit(t[r], values[r], 1)[0] for r in (before, after)]
    assert weights @ values[rows] == pytest.approx(slope[1] - slope[0], rel=1e-9)
    assert slope_weights(t, after) @ values[after] == pytest.approx(slope[1], rel=1e-9)


def test_the_lag_correlation_is_toeplitz_and_zero_beyond_the_last_lag() -> None:
    rows = np.array([0, 1, 2, 5, 9])
    table = lag_correlation(rows, (0.6, 0.3))
    assert table[0, 1] == table[1, 2] == 0.6 and table[0, 2] == 0.3
    assert table[2, 3] == table[3, 4] == 0.0 and np.all(np.diag(table) == 1.0)
    assert np.array_equal(lag_correlation(rows, ()), np.eye(rows.size))


def test_the_covariance_matches_a_brute_force_double_sum() -> None:
    """w' C w summed term by term over every pair of rows of the two windows."""
    u, v, s = body()
    d = descriptor(u, v, s)
    pixel, relative = 2.0, 0.07
    rows, weights = change_weights(times(), *windows())
    brute = np.zeros((3, 3))
    for i, row_i in enumerate(rows):
        for j, row_j in enumerate(rows):
            lag = abs(int(row_i) - int(row_j))
            term = relative**2 * lag_value(CORRELATED[1], lag) * np.outer(d[row_i], d[row_j])
            pixel_term = pixel**2 * lag_value(CORRELATED[0], lag) / (s[row_i] * s[row_j])
            term[:2, :2] += pixel_term * np.eye(2)
            brute += weights[i] * weights[j] * term
    exact = covariance(d, s, pixel, relative, CORRELATED)
    assert exact == pytest.approx(brute, rel=1e-10, abs=1e-14)


def test_the_exact_covariance_matches_seeded_draws() -> None:
    """The (3, 3) covariance of the rate change over ``NULL_DRAWS`` seeded draws of the
    first-order noise (shared, autocorrelated size noise) is within ``COVARIANCE_TOLERANCE`` of
    w' C w. (Reading s itself with noise adds about 8 (sigma_s / s)^2 to the variance: second
    order, left out of the covariance.)"""
    rng = np.random.default_rng(20261107)
    pixel, relative = 3.0, 0.10
    drawn = np.cov(changes(first_order(rng, pixel, relative, CORRELATED, NULL_DRAWS)), rowvar=False)
    u, v, s = body()
    exact = covariance(descriptor(u, v, s), s, pixel, relative, CORRELATED)
    scale = np.sqrt(np.outer(np.diag(exact), np.diag(exact)))
    assert np.all(np.abs(drawn - exact) <= COVARIANCE_TOLERANCE * scale)
    assert abs(exact[0, 2]) > 0.1 * scale[0, 2]


@pytest.mark.parametrize("relative", [0.01, 0.05, 0.10])
@pytest.mark.parametrize("lags", [WHITE, CORRELATED], ids=["white", "correlated"])
def test_without_an_impulse_the_evidence_is_uniform(relative: float, lags: Lags) -> None:
    """A constant-velocity body (no impulse) read with shared relative size noise: p =
    F_chi2_3(q), q against the covariance at the noisy read, is uniform."""
    rng = np.random.default_rng(20261108)
    pixel = 3.0
    descriptors = seen(rng, pixel, relative, lags, NULL_DRAWS)
    evidences = np.empty(NULL_DRAWS)
    for n, (d, change) in enumerate(zip(descriptors, changes(descriptors), strict=True)):
        cov = covariance(d, FOCAL / d[:, 2], pixel, relative, lags)
        evidences[n] = evidence(impulse_statistic(change, cov))
    assert abs(float(np.mean(evidences > 0.9)) - 0.1) <= TAIL_TOLERANCE
    assert abs(float(np.mean(evidences < 0.1)) - 0.1) <= TAIL_TOLERANCE
    assert stats.kstest(evidences, "uniform").pvalue > KS_FLOOR


def test_exact_inputs_have_zero_sigma() -> None:
    u, v, s = body()
    assert not covariance(descriptor(u, v, s), s, 0.0, 0.0, CORRELATED).any()
    flip = flip_probability(MARGINS, np.zeros(MARGINS.size), None)
    assert count_sigma(flip, FRAMES_PER_ROW) == 0.0
    assert magnitude_sigma(14, 0.2, 0.0, 0.0) == 0.0


@pytest.mark.parametrize("factor", [0.5, 2.0, 7.0])
def test_the_rate_change_sigma_is_linear_in_the_row_sigma(factor: float) -> None:
    u, v, s = body()
    d = descriptor(u, v, s)
    base = np.sqrt(np.diag(covariance(d, s, 1.5, 0.0, CORRELATED)))
    scaled = np.sqrt(np.diag(covariance(d, s, 1.5 * factor, 0.0, CORRELATED)))
    assert scaled[:2] == pytest.approx(factor * base[:2], rel=1e-12)
    assert not scaled[2]


def test_size_noise_alone_is_rank_one_and_not_positive_definite() -> None:
    rows, weights = change_weights(times(), *windows())
    noise = (np.full(ROWS, 80.0), np.zeros(ROWS), np.full(ROWS, 0.1))
    single = change_covariance(rows, weights, np.ones((ROWS, 3)), noise, (), ())
    assert np.linalg.matrix_rank(single) == 1 and not positive_definite(single)
    u, v, s = body()
    assert positive_definite(covariance(descriptor(u, v, s), s, 1.0, 0.1))


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


def test_the_sigma_of_s_matches_seeded_draws() -> None:
    """A pass-through (no impulse) of two constant-velocity bodies read with the noise model:
    the empirical spread of s over ``DRAWS`` draws is within ``TOLERANCE`` of the root mean
    square sigma the check reports on those same draws."""
    rng = np.random.default_rng(20261106)
    pixel, relative = 3.0, 0.10
    bodies = [seen(rng, pixel, relative, CORRELATED, DRAWS) for _ in range(2)]
    row_sigma = np.ones(MARGINS.size)
    s, reported = np.empty(DRAWS), np.empty(DRAWS)
    for draw in range(DRAWS):
        statistics = []
        for stack in bodies:
            d = stack[draw]
            cov = covariance(d, FOCAL / d[:, 2], pixel, relative, CORRELATED)
            statistics.append(impulse_statistic(changes(d[None])[0], cov))
        impulse = max(evidence(q) for q in statistics)
        margins = MARGINS + rng.normal(0.0, 1.0, MARGINS.size)
        frames = FRAMES_PER_ROW * int(np.sum(margins <= 0.0))
        frames_sigma = count_sigma(flip_probability(margins, row_sigma, None), FRAMES_PER_ROW)
        s[draw] = frames * (1.0 - impulse)
        reported[draw] = magnitude_sigma(frames, impulse, frames_sigma, evidence_sigma(statistics))
    expected = math.sqrt(float(np.mean(reported**2)))
    assert abs(float(np.std(s)) - expected) <= TOLERANCE * expected
