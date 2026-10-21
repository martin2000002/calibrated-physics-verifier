"""AC9: error entries as numbers and draws, and Monte Carlo propagation."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest
from support import entry

from physverify.physics.uncertainty import (
    MAD_TO_SIGMA,
    entry_draws,
    entry_rms,
    monte_carlo,
    robust_scale,
)
from physverify.schema.error_profile import ErrorEntry
from physverify.schema.stages import Stage

RESIDUALS = (-0.3, -0.1, 0.0, 0.2, 0.4)


def test_entry_rms() -> None:
    assert math.isnan(entry_rms(None))
    assert entry_rms(ErrorEntry.exact(Stage.DEPTH, "ground_truth", "depth_rel")) == 0.0
    assert entry_rms(entry(Stage.DEPTH, "vda_small", "depth_rel", 0.4, bias=0.3)) == 0.5


def test_entry_draws_bootstrap_the_residuals_reproducibly() -> None:
    residual_entry = entry(Stage.TRACK, "edgetam", "centroid_px_rel", 0.1, residuals=RESIDUALS)
    first = entry_draws(residual_entry, np.random.default_rng(3), (200, 2))
    again = entry_draws(residual_entry, np.random.default_rng(3), (200, 2))
    assert first is not None and again is not None
    assert first.shape == (200, 2)
    np.testing.assert_array_equal(first, again)
    assert set(np.unique(first).tolist()) <= set(RESIDUALS)


def test_entry_draws_are_normal_without_residuals() -> None:
    draws = entry_draws(
        entry(Stage.DEPTH, "vda_small", "depth_rel", 0.2, bias=0.05),
        np.random.default_rng(0),
        (20000,),
    )
    assert draws is not None
    assert float(draws.mean()) == pytest.approx(0.05, abs=0.01)
    assert float(draws.std()) == pytest.approx(0.2, rel=0.03)
    assert entry_draws(None, np.random.default_rng(0), (3,)) is None
    exact = entry_draws(
        ErrorEntry.exact(Stage.DEPTH, "gt", "depth_rel"), np.random.default_rng(0), (3,)
    )
    assert exact is not None and bool((exact == 0.0).all())


def test_monte_carlo_of_a_linear_model_matches_the_analytic_moments() -> None:
    rng = np.random.default_rng(11)
    draws = {"a": rng.normal(1.0, 0.2, 8000), "b": rng.normal(-2.0, 0.5, 8000)}

    def model(d: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.column_stack([3.0 * d["a"] + d["b"], d["a"] - 2.0 * d["b"]])

    result = monte_carlo(model, draws)
    expected_mean = np.array([1.0, 5.0])
    expected_std = np.array([math.hypot(0.6, 0.5), math.hypot(0.2, 1.0)])
    np.testing.assert_allclose(result.mean, expected_mean, rtol=0.03)
    np.testing.assert_allclose(result.std, expected_std, rtol=0.03)
    assert result.n_draws == 8000


def test_an_unknown_draw_makes_the_monte_carlo_sigma_unknown() -> None:
    draws = {"a": np.zeros(10), "b": None}
    result = monte_carlo(lambda d: np.column_stack([d["a"] + d["b"]]), draws)
    assert bool(np.isnan(result.std).all())


def test_robust_scale_is_the_normal_sigma_and_ignores_outliers() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(3.0, 2.0, 20000)
    assert robust_scale(sample) == pytest.approx(2.0, rel=0.03)
    spiked = np.concatenate([sample, np.full(500, 1e6), [np.nan, np.inf]])
    assert robust_scale(spiked) == pytest.approx(2.0, rel=0.05)
    assert math.isnan(robust_scale([np.nan])) and math.isnan(robust_scale([]))
    assert robust_scale([1.0, 1.0, 1.0]) == 0.0
    assert MAD_TO_SIGMA == 0.6745
