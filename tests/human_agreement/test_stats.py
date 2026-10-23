"""Agreement statistics against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binomtest

from physverify.human_agreement.policy_metrics import best_cut, h2_verdict, predict_cut
from physverify.human_agreement.stats import (
    accuracy,
    as_labels,
    auc,
    balanced_accuracy,
    caption_bootstrap,
    clustered_mcnemar,
    cohen_kappa,
    design_effect,
    f1_positive,
    mcnemar_exact,
    mde_accuracy,
)

Y = as_labels([True, True, True, False, False, False, False, False])
P = as_labels([True, True, False, True, False, False, False, False])
EXACT = 1e-12  # Closed-form quantities must match to this absolute tolerance


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


def test_the_agreement_verdict_needs_the_calibrated_rule_and_both_conditions() -> None:
    primary = {
        "mcnemar": {"p_value": 0.01},
        "verifier": {"accuracy": 0.8},
        "vlm": {"accuracy": 0.7},
        "bootstrap": {"intervals": {"difference.kappa": [0.02, 0.3]}},
        "mde_accuracy": 0.05,
    }
    assert h2_verdict(primary, True, 0.05).supported is True
    provisional = h2_verdict(primary, False, 0.05)
    assert provisional.supported is None and "no human-agreement verdict" in provisional.reason
    weak = {**primary, "bootstrap": {"intervals": {"difference.kappa": [-0.01, 0.3]}}}
    assert h2_verdict(weak, True, 0.05).supported is False


def test_the_agreement_verdict_with_a_gate_is_not_evaluable_whatever_the_numbers() -> None:
    primary = {
        "verifier": {"accuracy": 0.9},
        "vlm": {"accuracy": 0.5},
        "mcnemar": {"p_value": 0.001},
        "bootstrap": {"intervals": {"difference.kappa": [0.02, 0.3]}},
    }
    gated = h2_verdict(primary, False, 0.05, gate="aggregate insufficient")
    assert gated.status == "not_evaluable" and gated.applicable is False
    assert gated.supported is None and "aggregate insufficient" in gated.reason
    assert h2_verdict(primary, True, 0.05).status == "evaluated"
    assert h2_verdict(primary, False, 0.05).status == "pipeline_test"


def test_singleton_clusters_have_no_design_effect() -> None:
    effect = design_effect(["a", "b", "c", "d"], [1, 0, 1, 0])
    assert effect.deff == 1.0 and effect.icc == 0.0 and effect.clusters == 4


def test_fewer_than_two_clusters_give_nan() -> None:
    assert math.isnan(design_effect(["a", "a"], [1, 0]).deff)


def test_the_anova_icc_matches_the_hand_computation() -> None:
    # Two captions of 3 items: means 1 and 1/3, grand 2/3; MSB = 3 * 2 * (1/3)^2 / 1 = 2/3,
    # MSW = (0 + 2/3) / 4 = 1/6, m0 = 3, ICC = (2/3 - 1/6) / (2/3 + 2/6) = 1/2, deff = 1 + 2/2.
    effect = design_effect(["a"] * 3 + ["b"] * 3, [1, 1, 1, 1, 0, 0])
    assert effect.icc == pytest.approx(0.5, abs=EXACT)
    assert effect.deff == pytest.approx(2.0, abs=EXACT)


@pytest.mark.parametrize("seed", range(5))
def test_the_clustered_mde_is_never_below_the_independent_one(seed: int) -> None:
    rng = np.random.default_rng(seed)
    clusters = [str(c) for c in rng.integers(0, 12, size=80)]
    indicator = rng.integers(0, 2, size=80)
    effect = design_effect(clusters, indicator)
    assert effect.deff >= 1.0
    d = float(np.mean(indicator))
    plain = mde_accuracy(80, d, 0.05, 0.8)
    clustered = mde_accuracy(80 / effect.deff, d, 0.05, 0.8)
    assert math.isnan(clustered) or clustered >= plain


def _paired_labels(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return tuple(rng.integers(0, 2, n) for _ in range(3))  # type: ignore[return-value]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_caption_mcnemar_with_one_caption_per_clip_is_the_exact_mcnemar(seed: int) -> None:
    y, a, b = _paired_labels(seed, 30)
    captions = [f"c{i}" for i in range(y.size)]
    found = clustered_mcnemar(y, a, b, captions, exact_max=40, n=0, seed=0)
    assert found.method == "exact"
    assert found.p_value == pytest.approx(mcnemar_exact(y, a, b).p_value, abs=1e-12)


def test_the_caption_mcnemar_sums_within_a_caption() -> None:
    y = np.asarray([1, 1, 1, 1])
    a = np.asarray([1, 1, 0, 1])
    b = np.asarray([0, 0, 1, 1])
    found = clustered_mcnemar(y, a, b, ["x", "x", "z", "z"], exact_max=5, n=0, seed=0)
    assert (found.statistic, found.discordant_captions) == (1, 2)
    assert found.p_value == pytest.approx(1.0)


def test_the_monte_carlo_caption_mcnemar_adds_one_and_tracks_the_exact() -> None:
    y, a, b = _paired_labels(7, 60)
    captions = [f"c{i // 2}" for i in range(y.size)]
    exact = clustered_mcnemar(y, a, b, captions, exact_max=60, n=0, seed=0)
    sampled = clustered_mcnemar(y, a, b, captions, exact_max=0, n=4000, seed=3)
    assert sampled.method == "monte_carlo" and sampled.replicas == 4000
    assert sampled.p_value >= 1 / 4001
    assert sampled.p_value == pytest.approx(exact.p_value, abs=0.03)
    assert math.isnan(clustered_mcnemar(y, a, b, captions, exact_max=0, n=0, seed=0).p_value)
