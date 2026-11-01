import numpy as np
import pytest

from physverify.physics.measures.gravity_fit import fall_fit
from physverify.physics.measures.weighted_fall import (
    WeightedFallFit,
    accel_wald,
    wald_terms,
    weighted_fall_fit,
)

CHI2_3_Q99 = 11.344866730144373
T = np.linspace(0.0, 1.0, 13)
A = np.array([0.4, -0.2, -9.8])


def fall(t: np.ndarray, a: np.ndarray = A) -> np.ndarray:
    tau = t - t[0]
    p0, v0 = np.array([0.1, 0.2, 1.0]), np.array([0.5, 0.0, 3.0])
    return np.asarray(p0 + np.outer(tau, v0) + 0.5 * np.outer(tau**2, a))


def sigma_rows(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.001, 0.02, size=(n, 3))


def fitted(t: np.ndarray, positions: np.ndarray, sigma: np.ndarray) -> WeightedFallFit:
    fit = weighted_fall_fit(t, positions, sigma)
    assert fit is not None
    return fit


def test_an_exact_parabola_gives_its_acceleration_and_no_contrast() -> None:
    first = fitted(T, fall(T), sigma_rows(T.size))
    second = fitted(T + 1.0, fall(T + 1.0), sigma_rows(T.size, seed=1))
    np.testing.assert_allclose(first.a, A, atol=1e-9)
    assert accel_wald(first, second) == pytest.approx(0.0, abs=1e-12)


def test_equal_sigma_is_least_squares_with_its_textbook_variance() -> None:
    noisy = fall(T) + np.random.default_rng(2).normal(0.0, 0.01, size=(T.size, 3))
    sigma = 0.01
    fit = fitted(T, noisy, np.full((T.size, 3), sigma))
    np.testing.assert_allclose(fit.a, fall_fit(T, noisy).a, atol=1e-9)
    design = np.column_stack([np.ones_like(T), T, 0.5 * T**2])
    expected = sigma**2 * np.linalg.inv(design.T @ design)[2, 2]
    np.testing.assert_allclose(fit.a_var, np.full(3, expected), rtol=1e-9)


def test_the_contrast_is_chi_square_with_three_degrees_under_the_null() -> None:
    rng = np.random.default_rng(3)
    late = T + 1.2
    s1, s2 = sigma_rows(T.size, seed=4), sigma_rows(T.size, seed=5)
    values = np.array(
        [
            accel_wald(
                fitted(T, fall(T) + rng.normal(0.0, s1), s1),
                fitted(late, fall(late) + rng.normal(0.0, s2), s2),
            )
            for _ in range(2000)
        ]
    )
    assert values.mean() == pytest.approx(3.0, abs=0.3)
    assert np.mean(values > CHI2_3_Q99) == pytest.approx(0.01, abs=0.008)


def test_the_contrast_sums_the_axes_and_scales_with_sigma() -> None:
    sigma = sigma_rows(T.size)
    first = fitted(T, fall(T), sigma)
    second = fitted(T, fall(T, A * 1.1), sigma)
    assert accel_wald(first, second) == pytest.approx(float(np.sum(wald_terms(first, second))))
    wider = [fitted(T, fall(T, a), 2.0 * sigma) for a in (A, A * 1.1)]
    assert accel_wald(*wider) == pytest.approx(accel_wald(first, second) / 4.0)


def test_rows_without_position_or_sigma_are_left_out() -> None:
    positions, sigma = fall(T), sigma_rows(T.size)
    positions[2] = np.nan
    sigma[5] = np.nan
    sigma[7] = 0.0
    np.testing.assert_allclose(fitted(T, positions, sigma).a, A, atol=1e-9)


def test_fewer_than_three_usable_rows_give_no_fit() -> None:
    sigma = sigma_rows(T.size)
    sigma[2:] = np.nan
    assert weighted_fall_fit(T, fall(T), sigma) is None
