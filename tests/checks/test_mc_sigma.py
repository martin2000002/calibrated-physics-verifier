import math

import numpy as np
import pytest

from physverify.checks._uncertainty import mc_sigma


def total(noise: dict[str, np.ndarray]) -> np.ndarray:
    return sum(value.reshape(value.shape[0], -1).sum(axis=1) for value in noise.values())


def test_exact_inputs_and_point_intervals_give_zero() -> None:
    rng = np.random.default_rng(0)
    assert mc_sigma(total, {"x": np.zeros(3)}, 64, rng) == 0.0
    assert mc_sigma(total, {"x": np.zeros(3)}, 64, rng, {"mu": (0.5, 0.5)}) == 0.0


def test_a_uniform_parameter_has_the_uniform_std() -> None:
    rng = np.random.default_rng(1)
    sigma = mc_sigma(total, {"x": np.zeros(3)}, 20000, rng, {"mu": (0.3, 1.0)})
    assert sigma == pytest.approx(0.7 / math.sqrt(12.0), rel=0.03)


def test_the_uniform_draws_stay_inside_the_interval() -> None:
    seen: list[np.ndarray] = []

    def keep(noise: dict[str, np.ndarray]) -> np.ndarray:
        seen.append(noise["mu"])
        return noise["mu"]

    mc_sigma(keep, {"x": np.zeros(1)}, 500, np.random.default_rng(2), {"mu": (0.3, 1.0)})
    assert seen[0].shape == (500,)
    assert seen[0].min() >= 0.3 and seen[0].max() <= 1.0


def test_the_gaussian_draws_do_not_move_with_a_uniform_key() -> None:
    sigmas = {"x": np.full(2, 0.1)}
    plain = mc_sigma(total, sigmas, 64, np.random.default_rng(3))
    first = mc_sigma(total, sigmas, 64, np.random.default_rng(3), {"mu": (0.0, 0.0)})
    assert plain == first


def test_an_unknown_sigma_stays_nan_with_a_uniform_key() -> None:
    sigmas = {"x": np.full(2, math.nan)}
    assert math.isnan(mc_sigma(total, sigmas, 64, np.random.default_rng(4), {"mu": (0.3, 1.0)}))
