"""Shared statistics: exact intervals, conformal rank, scene folds, scene bootstrap, AUROC, ECE."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binomtest, mannwhitneyu

from physverify.calibration.stats import (
    SceneBlock,
    auroc,
    balanced_weights,
    clopper_pearson,
    clopper_pearson_lower,
    clustered_rate,
    conformal_index,
    conformal_quantile,
    ece,
    fit_logistic,
    grouped_folds,
    mad,
    profile_interval,
    scene_bootstrap,
)
from physverify.physics.uncertainty import robust_scale


@pytest.mark.parametrize(("k", "n"), [(0, 20), (3, 20), (20, 20), (1, 100), (7, 13)])
def test_clopper_pearson_is_the_exact_binomial_interval(k: int, n: int) -> None:
    expected = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    found = clopper_pearson(k, n, 0.95)
    assert found.low == pytest.approx(expected.low, abs=1e-12)
    assert found.high == pytest.approx(expected.high, abs=1e-12)


def test_zero_out_of_twenty_has_upper_bound_0_168() -> None:
    found = clopper_pearson(0, 20, 0.95)
    assert found.low == 0.0
    assert found.high == pytest.approx(0.168, abs=5e-4)
    assert clopper_pearson(0, 0, 0.95).low is None


def test_conformal_rank_needs_19_scenes_at_alpha_005() -> None:
    assert conformal_index(19, 0.05) == 19
    assert conformal_index(100, 0.05) == 96
    full = conformal_quantile(np.arange(1.0, 20.0), 0.05)
    assert (full.value, full.sufficient) == (19.0, True)
    short = conformal_quantile(np.arange(1.0, 19.0), 0.05)
    assert (short.value, short.sufficient) == (18.0, False)
    assert conformal_quantile([math.nan], 0.05).value is None


def test_conformal_counts_nan_and_ranks_infinity_as_the_largest() -> None:
    """calibration.md #8: a NaN is dropped and counted; +inf is not silently dropped."""
    values = [*np.arange(1.0, 20.0), math.nan]
    found = conformal_quantile(values, 0.05)
    assert (found.value, found.n, found.n_dropped) == (19.0, 19, 1)
    with_inf = conformal_quantile([*np.arange(1.0, 19.0), math.inf], 0.05)
    assert (with_inf.value, with_inf.sufficient, with_inf.n) == (None, False, 19)
    below = conformal_quantile([*np.arange(1.0, 40.0), math.inf], 0.05)
    assert below.value == 39.0 and below.sufficient


@pytest.mark.parametrize(("k", "n", "low"), [(5, 5, 0.05 ** (1 / 5)), (20, 20, 0.05 ** (1 / 20))])
def test_the_one_sided_lower_bound_of_perfect_rates(k: int, n: int, low: float) -> None:
    assert clopper_pearson_lower(k, n, 0.95) == pytest.approx(low)
    assert clopper_pearson_lower(0, 0, 0.95) == 0.0


def test_folds_never_split_a_scene() -> None:
    scenes = [f"s{i % 7}" for i in range(30)]
    folds = grouped_folds(scenes, 5, seed=3)
    assert len(folds) == 5
    assert set().union(*folds) == set(scenes)
    assert sum(len(f) for f in folds) == 7
    assert grouped_folds(["a", "b"], 5, seed=0) == grouped_folds(["b", "a"], 5, seed=0)
    assert len(grouped_folds(["a", "b"], 5, seed=0)) == 2


def test_the_bootstrap_resamples_whole_scenes() -> None:
    blocks = [SceneBlock("A", 1), SceneBlock("A", 1), SceneBlock("B", 0), SceneBlock("C", 0)]
    ci = scene_bootstrap(blocks, lambda s: {"odd": float(sum(s) % 2)}, 300, 0, 0.95)
    assert (ci["odd"].low, ci["odd"].high) == (0.0, 0.0)


def test_auroc_is_mann_whitney() -> None:
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1.0, 1.0, 30), rng.normal(0.0, 1.0, 40)
    expected = mannwhitneyu(pos, neg).statistic / (30 * 40)
    assert auroc(pos, neg) == pytest.approx(expected)
    assert auroc([1.0, 2.0], [0.0]) == 1.0
    assert auroc([1.0], [1.0]) == 0.5
    assert math.isnan(auroc([], [1.0]))


def test_ece_on_a_hand_made_case() -> None:
    assert ece([0.2, 0.2, 0.8, 0.8], [0, 1, 1, 1]) == pytest.approx(0.25)
    assert ece([0.2, 0.8], [0, 1], weights=[3.0, 1.0]) == pytest.approx(0.2)


def test_the_penalised_logistic_is_finite_under_separation_and_can_be_monotone() -> None:
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    fit = fit_logistic(x, [0, 0, 1, 1], l2=0.01)
    assert fit is not None and math.isfinite(fit.b) and fit.b > 0
    reversed_fit = fit_logistic(x, [1, 1, 0, 0], l2=0.01, monotone=True)
    assert reversed_fit is not None and reversed_fit.b == 0.0
    assert fit_logistic(x, [1, 1, 1, 1], l2=0.01) is None


def test_balanced_weights_give_each_class_half_the_mass() -> None:
    w = balanced_weights([1, 0, 0, 0])
    assert w[0] == pytest.approx(w[1:].sum())
    assert mad([1.0, 1.0, 1.0]) == 1.0


def test_one_flag_per_scene_is_clopper_pearson() -> None:
    found = clustered_rate([(f"s{i}", i < 3) for i in range(20)], 200, 0, 0.95)
    exact = clopper_pearson(3, 20, 0.95)
    assert (found.ci_method, found.n_scenes, found.ci.low, found.ci.high) == (
        "clopper_pearson",
        20,
        exact.low,
        exact.high,
    )


def test_clustered_flags_use_the_scenes() -> None:
    flags = [(f"s{s}", s < 4) for s in range(10) for _ in range(5)]
    found = clustered_rate(flags, 300, 0, 0.95)
    assert (found.k, found.n, found.n_scenes, found.ci_method) == (20, 50, 10, "scene_bootstrap")
    assert found.ci.low is not None and found.ci.high is not None
    assert found.ci.low < 0.4 < found.ci.high
    none = clustered_rate([(f"s{s}", False) for s in range(10) for _ in range(5)], 300, 0, 0.95)
    assert none.ci_method == "clopper_pearson_scenes"
    assert none.ci.high == pytest.approx(clopper_pearson(0, 10, 0.95).high)


def test_the_profile_interval_contains_the_maximum_and_can_stay_open() -> None:
    rng = np.random.default_rng(0)
    x = np.repeat(np.log([0.05, 0.1, 0.2, 0.4]), 25)
    y = (rng.random(x.size) < 1.0 / (1.0 + np.exp(-(4.0 * (x - math.log(0.1)) + 2.94)))).astype(
        float
    )
    bounds = (math.log(0.025), math.log(0.8))
    found = profile_interval(x, y, 2.94, 1e-3, 0.95, bounds)
    assert found.low is not None and found.high is not None
    assert bounds[0] < found.low < math.log(0.1) < found.high < bounds[1]
    flat = profile_interval(x, np.zeros_like(x), 2.94, 1e-3, 0.95, bounds)
    assert flat.high is None


def test_the_calibration_mad_calls_the_robust_scale() -> None:
    values = [0.0, 1.0, 2.0, 10.0]
    assert mad(values) == robust_scale(values)
    assert mad([]) == 1.0 and mad([5.0, 5.0]) == 1.0
