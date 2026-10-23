from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from physverify.generator_comparison.settings import ComparisonSettings
from physverify.generator_comparison.stats import (
    Diff,
    TestResult,
    hodges_lehmann,
    holm,
    mde_paired,
    mean_interval,
    method_label,
    paired_test,
    sample_sd,
    significant_negative,
    supported,
    wilcoxon_p,
)


def _result(p: float | None, hl: float | None) -> TestResult:
    return TestResult(n=5, p=p, hl=hl, mean=hl, ci_lo=None, ci_hi=None, method="m")


def test_wilcoxon_is_undefined_without_signal(comparison_cfg: ComparisonSettings) -> None:
    assert wilcoxon_p([0.0, 0.0, 0.0], comparison_cfg) is None
    assert wilcoxon_p([0.3], comparison_cfg) is None


def test_wilcoxon_is_small_for_a_consistent_shift(comparison_cfg: ComparisonSettings) -> None:
    p = wilcoxon_p([0.1, 0.2, 0.15, 0.3, 0.25, 0.12, 0.18, 0.22], comparison_cfg)
    assert p is not None
    assert p < 0.05


def test_wilcoxon_method_is_pinned(comparison_cfg: ComparisonSettings) -> None:
    assert method_label(comparison_cfg) == "wilcoxon:asymptotic:pratt:no-correction"
    data = comparison_cfg.model_dump()
    with pytest.raises(ValidationError):
        ComparisonSettings.model_validate({**data, "wilcoxon_method": "exact"})
    with pytest.raises(ValidationError):
        ComparisonSettings.model_validate({**data, "zero_method": "wilcox"})


def test_hodges_lehmann_is_the_median_of_walsh_averages() -> None:
    assert hodges_lehmann([1.0, 2.0, 3.0]) == 2.0
    assert hodges_lehmann([0.0, 10.0]) == 5.0
    assert hodges_lehmann([]) is None


def test_holm_counts_undefined_p_in_m() -> None:
    adjusted = holm({"a": 0.01, "b": 0.04, "c": None})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.08)
    assert adjusted["c"] is None


def test_holm_is_monotone_and_capped() -> None:
    adjusted = holm({"a": 0.5, "b": 0.02})
    assert adjusted == {"a": pytest.approx(0.5), "b": pytest.approx(0.04)}
    assert holm({"a": 0.9, "b": 0.8})["a"] == 1.0


def test_mean_interval_resamples_units_and_is_reproducible(
    comparison_cfg: ComparisonSettings,
) -> None:
    diffs = [Diff(f"p{i}", v) for i, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])]
    lo, hi = mean_interval(diffs, comparison_cfg)
    assert lo is not None and hi is not None
    assert lo <= 0.3 <= hi
    assert mean_interval(diffs, comparison_cfg) == (lo, hi)
    assert mean_interval(diffs[:1], comparison_cfg) == (None, None)


def test_one_cluster_has_a_degenerate_interval(comparison_cfg: ComparisonSettings) -> None:
    lo, hi = mean_interval([Diff("t0", 0.1), Diff("t0", 0.3)], comparison_cfg)
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)


def test_paired_test_reports_everything(comparison_cfg: ComparisonSettings) -> None:
    diffs = [Diff(f"p{i}", v) for i, v in enumerate([0.1, -0.05, 0.2, 0.3])]
    result = paired_test(diffs, comparison_cfg, dropped=2)
    assert (result.n, result.dropped) == (4, 2)
    assert result.mean == pytest.approx(0.1375)
    assert result.method == method_label(comparison_cfg)


def test_supported_needs_significance_and_a_positive_shift() -> None:
    assert supported(_result(0.01, 0.2), 0.05)
    assert not supported(_result(0.01, -0.2), 0.05)
    assert not supported(_result(0.2, 0.2), 0.05)
    assert not supported(_result(None, 0.2), 0.05)
    assert not supported(None, 0.05)
    assert significant_negative(_result(0.01, -0.2), 0.05)
    assert not significant_negative(_result(0.01, 0.2), 0.05)


def test_mde_shrinks_with_n_and_scales_with_sd() -> None:
    small, large = mde_paired(10, 0.2, 0.05, 0.8), mde_paired(40, 0.2, 0.05, 0.8)
    assert small is not None and large is not None
    assert large == pytest.approx(small / 2)
    expected = (1.959964 + 0.841621) / math.sqrt(10) / math.sqrt(3 / math.pi) * 0.2
    assert small == pytest.approx(expected, rel=1e-5)
    assert mde_paired(0, 0.2, 0.05, 0.8) is None
    assert mde_paired(10, math.nan, 0.05, 0.8) is None


def test_sample_sd_needs_two_values() -> None:
    assert sample_sd([1.0]) is None
    assert sample_sd([1.0, 3.0]) == pytest.approx(math.sqrt(2))
