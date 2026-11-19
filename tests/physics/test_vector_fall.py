import numpy as np
import pytest

from physverify.physics.line_of_sight import line_of_sight_covariance
from physverify.physics.measures.gravity_fit import fall_fit
from physverify.physics.measures.vector_fall import (
    VectorFallFit,
    gls_ballistic_fit,
    gls_fall_fit,
    tilt_statistic,
)
from physverify.physics.measures.weighted_fall import weighted_fall_fit
from physverify.schema.states import CoordFrame

T = np.linspace(0.0, 1.0, 13)
DOWN = np.array([0.0, 0.6, 0.8])
A = 9.8 * DOWN


def fall(t: np.ndarray, a: np.ndarray = A) -> np.ndarray:
    tau = t - t[0]
    p0, v0 = np.array([0.1, -0.3, 3.0]), np.array([0.5, -2.0, 1.0])
    return np.asarray(p0 + np.outer(tau, v0) + 0.5 * np.outer(tau**2, a))


def diagonal(sigma: np.ndarray) -> np.ndarray:
    return np.einsum("nk,kl->nkl", sigma**2, np.eye(3))


def fitted(t: np.ndarray, positions: np.ndarray, cov: np.ndarray) -> VectorFallFit:
    fit = gls_fall_fit(t, positions, cov)
    assert fit is not None
    return fit


def lifted(t: np.ndarray, rel: float, floor: float = 0.003) -> tuple[np.ndarray, np.ndarray]:
    """Sigma of a monocular lift: a relative depth error plus a pixel floor on x and y."""
    p = fall(t)
    sigma = np.column_stack(
        [np.hypot(rel * p[:, 0], floor), np.hypot(rel * p[:, 1], floor), rel * p[:, 2]]
    )
    cov = line_of_sight_covariance(p, sigma, sigma == 0.0, CoordFrame.CAMERA_OPENCV, 1e-6)
    return p, cov


def test_an_exact_parabola_gives_its_acceleration() -> None:
    _, cov = lifted(T, 0.02)
    fit = fitted(T, fall(T), cov)
    np.testing.assert_allclose(fit.a, A, atol=1e-9)
    assert tilt_statistic(fit, DOWN, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_an_isotropic_covariance_is_least_squares() -> None:
    noisy = fall(T) + np.random.default_rng(0).normal(0.0, 0.01, (T.size, 3))
    fit = fitted(T, noisy, diagonal(np.full((T.size, 3), 0.01)))
    np.testing.assert_allclose(fit.a, fall_fit(T, noisy).a, atol=1e-9)


def test_a_diagonal_covariance_is_the_per_axis_weighted_fit() -> None:
    rng = np.random.default_rng(1)
    sigma = rng.uniform(0.001, 0.02, (T.size, 3))
    noisy = fall(T) + rng.normal(0.0, 1.0, (T.size, 3)) * sigma
    fit = fitted(T, noisy, diagonal(sigma))
    weighted = weighted_fall_fit(T, noisy, sigma)
    assert weighted is not None
    np.testing.assert_allclose(fit.a, weighted.a, rtol=1e-9)
    np.testing.assert_allclose(np.diag(fit.a_cov), weighted.a_var, rtol=1e-9)


def test_the_ballistic_fit_recovers_g_along_the_given_direction() -> None:
    _, cov = lifted(T, 0.02)
    fit = gls_ballistic_fit(T, fall(T), cov, DOWN)
    assert fit is not None
    assert fit.g == pytest.approx(9.8, rel=1e-9)
    assert fit.g_var > 0.0


def test_too_few_distinct_times_give_no_fit() -> None:
    _, cov = lifted(T[:3], 0.02)
    times = np.array([0.0, 0.0, 0.1])
    assert gls_fall_fit(times, fall(T[:3]), cov) is None
    assert gls_ballistic_fit(times, fall(T[:3]), cov, DOWN) is None


def test_the_tilt_is_chi_square_with_two_degrees_per_window_under_the_null() -> None:
    p, cov = lifted(T, 0.03)
    rng = np.random.default_rng(7)
    root = np.linalg.cholesky(cov)
    draws = [
        tilt_statistic(
            fitted(T, p + np.einsum("nkl,nl->nk", root, rng.normal(size=p.shape)), cov), DOWN, 0.0
        )
        for _ in range(2000)
    ]
    assert np.mean(draws) == pytest.approx(2.0, abs=0.15)


def test_noise_along_the_rays_does_not_move_the_tilt_much() -> None:
    p, cov = lifted(T, 0.03)
    breathing = p * (1.0 + 0.03 * np.sin(np.linspace(0.0, 6.0, T.size)))[:, None]
    noisy = p + np.random.default_rng(3).normal(0.0, 0.003, p.shape)
    assert tilt_statistic(fitted(T, breathing, cov), DOWN, 0.0) < 9.21
    assert tilt_statistic(fitted(T, noisy, cov), DOWN, 0.0) < 9.21


def test_a_tilted_acceleration_scores_and_a_gravity_sigma_absorbs_it() -> None:
    _, cov = lifted(T, 0.02)
    tilted = fall(T, 9.8 * np.array([0.3, 0.6, 0.74]))
    fit = fitted(T, tilted, cov)
    assert tilt_statistic(fit, DOWN, 0.0) > 100.0
    assert tilt_statistic(fit, DOWN, 0.0) > 10 * tilt_statistic(fit, DOWN, 0.3)
