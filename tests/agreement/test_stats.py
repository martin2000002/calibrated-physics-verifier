"""Agreement statistics against hand-computed values (C4)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binomtest

from physverify.agreement.analysis import best_cut, h2_verdict, predict_cut
from physverify.agreement.stats import (
    accuracy,
    as_labels,
    auc,
    balanced_accuracy,
    caption_bootstrap,
    cohen_kappa,
    f1_positive,
    mcnemar_exact,
    mde_accuracy,
)

Y = as_labels([True, True, True, False, False, False, False, False])
P = as_labels([True, True, False, True, False, False, False, False])
EXACT = 1e-12  # C4: closed-form quantities must match to this absolute tolerance


def test_classification_metrics_match_the_hand_computation() -> None:
    # tp 2, fn 1, fp 1, tn 4.
    assert accuracy(Y, P) == pytest.approx(6 / 8, abs=EXACT)
    assert balanced_accuracy(Y, P) == pytest.approx((2 / 3 + 4 / 5) / 2, abs=EXACT)
    assert f1_positive(Y, P) == pytest.approx(4 / 6, abs=EXACT)
    p_e = (3 / 8) * (3 / 8) + (5 / 8) * (5 / 8)
    assert cohen_kappa(Y, P) == pytest.approx((6 / 8 - p_e) / (1 - p_e), abs=EXACT)


def test_degenerate_cases_are_nan() -> None:
    empty = as_labels([])
    assert math.isnan(accuracy(empty, empty))
    ones = as_labels([True, True])
    assert math.isnan(cohen_kappa(ones, ones))
    assert math.isnan(balanced_accuracy(ones, ones))


def test_auc_counts_ties_as_half_and_drops_nan() -> None:
    y = as_labels([True, True, False, False])
    score = np.array([0.9, 0.5, 0.5, math.nan])
    assert auc(y, score) == pytest.approx((1 + 0.5) / 2, abs=EXACT)


def test_mcnemar_is_the_exact_binomial_test_on_discordant_pairs() -> None:
    y = as_labels([True] * 10)
    a = as_labels([True] * 8 + [False] * 2)
    b = as_labels([False] * 5 + [True] * 3 + [False, True])
    result = mcnemar_exact(y, a, b)
    assert (result.b, result.c) == (5, 1)
    assert result.p_value == pytest.approx(binomtest(5, 6, 0.5).pvalue, abs=EXACT)
    same = mcnemar_exact(y, a, a)
    assert (same.b, same.c, same.p_value) == (0, 0, 1.0)


def test_the_minimum_detectable_difference_shrinks_with_n() -> None:
    small, large = mde_accuracy(100, 0.3, 0.05, 0.8), mde_accuracy(1000, 0.3, 0.05, 0.8)
    assert 0 < large < small <= 0.3
    # n = (z_a sqrt(d) + z_b sqrt(d - delta^2))^2 / delta^2 at the solution.
    z_a, z_b = 1.959964, 0.841621
    needed = (z_a * math.sqrt(0.3) + z_b * math.sqrt(0.3 - large**2)) ** 2 / large**2
    assert needed == pytest.approx(1000, rel=1e-3)
    assert math.isnan(mde_accuracy(100, 0.0, 0.05, 0.8))


@pytest.mark.parametrize("n", [10, 50, 100, 563, 5000])
@pytest.mark.parametrize("d", [0.05, 0.1, 0.3, 0.5, 1.0])
def test_the_detectable_difference_never_exceeds_the_discordant_rate(n: int, d: float) -> None:
    mde = mde_accuracy(n, d, 0.05, 0.8)
    assert math.isnan(mde) or 0 < mde <= d


def test_an_undetectable_effect_is_nan() -> None:
    assert math.isnan(mde_accuracy(50, 0.05, 0.05, 0.8))
    assert math.isnan(mde_accuracy(10, 0.5, 0.05, 0.8))


def test_the_caption_bootstrap_is_seeded_and_resamples_captions() -> None:
    items = list(range(6))
    captions = ["a", "a", "b", "b", "c", "c"]

    def statistic(sample: list[int]) -> dict[str, float]:
        return {"mean": float(np.mean(sample))}

    first = caption_bootstrap(items, lambda i: captions[i], statistic, 200, 7, 0.9)
    again = caption_bootstrap(items, lambda i: captions[i], statistic, 200, 7, 0.9)
    assert first.intervals == again.intervals
    low, high = first.intervals["mean"].low, first.intervals["mean"].high
    assert 0.5 <= low <= 2.5 <= high <= 4.5


def test_the_caption_bootstrap_counts_replicas_where_kappa_is_nan() -> None:
    y = as_labels([True, True, False, False])
    p = as_labels([True, True, False, False])
    captions = ["a", "a", "b", "b"]

    def statistic(sample: list[int]) -> dict[str, float]:
        pick = np.asarray(sample)
        return {"kappa": cohen_kappa(y[pick], p[pick])}

    result = caption_bootstrap(list(range(4)), lambda i: captions[i], statistic, 100, 3, 0.9)
    rng = np.random.default_rng(3)
    one_caption = sum(len(set(rng.integers(0, 2, size=2))) == 1 for _ in range(100))
    assert result.non_finite == {"kappa": one_caption} and one_caption > 0
    assert result.replicas == 100


def test_best_cut_maximises_balanced_accuracy_and_breaks_ties_low() -> None:
    y = as_labels([False, False, True, True])
    assert best_cut(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(0.2)
    tied = as_labels([True, False])
    assert best_cut(tied, np.array([0.5, 0.5])) == pytest.approx(-0.5)
    assert best_cut(y, np.full(4, math.nan)) is None
    assert predict_cut(np.array([0.3, math.nan, 0.1]), 0.2).tolist() == [1, 0, 0]


def test_h2_needs_the_calibrated_rule_and_both_conditions() -> None:
    primary = {
        "mcnemar": {"p_value": 0.01},
        "verifier": {"accuracy": 0.8},
        "vlm": {"accuracy": 0.7},
        "bootstrap": {"intervals": {"difference.kappa": [0.02, 0.3]}},
        "mde_accuracy": 0.05,
    }
    assert h2_verdict(primary, True, 0.05)["supported"] is True
    provisional = h2_verdict(primary, False, 0.05)
    assert provisional["supported"] is None and "no H2 verdict" in provisional["reason"]
    weak = {**primary, "bootstrap": {"intervals": {"difference.kappa": [-0.01, 0.3]}}}
    assert h2_verdict(weak, True, 0.05)["supported"] is False
