"""Pre-registered statistics of H3 (ADR 0013 D3, D4, D6): Wilcoxon + Hodges-Lehmann, the
bootstrap interval of the mean, Holm and the minimum detectable effect.

The unit is the prompt: each ``Diff`` is one prompt's paired difference. Wilcoxon's method is
pinned in ``[h3]`` (asymptotic, Pratt zeros, no continuity correction), never chosen from the data.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import stats as sps

from physverify.core.models import FrozenModel
from physverify.generator_comparison.settings import ComparisonSettings
from physverify.stats.bootstrap import bootstrap

MEAN: Final = "mean"
# Asymptotic relative efficiency of the signed-rank test against the t test under normality.
WILCOXON_ARE: Final = 3.0 / math.pi


@dataclass(frozen=True)
class Diff:
    """One prompt's difference; ``clip_id`` is the resampled unit (prompt or template)."""

    clip_id: str
    value: float


class TestResult(FrozenModel):
    n: int
    dropped: int = 0
    p: float | None
    hl: float | None
    mean: float | None
    ci_lo: float | None
    ci_hi: float | None
    method: str


def supported(result: TestResult | None, alpha: float) -> bool:
    """Significant with a positive Hodges-Lehmann estimate (the estimand)."""
    return (
        result is not None
        and result.p is not None
        and result.hl is not None
        and result.p < alpha
        and result.hl > 0
    )


def significant_negative(result: TestResult | None, alpha: float) -> bool:
    return (
        result is not None
        and result.p is not None
        and result.hl is not None
        and result.p < alpha
        and result.hl < 0
    )


def method_label(cfg: ComparisonSettings) -> str:
    correction = "continuity" if cfg.correction else "no-correction"
    return f"wilcoxon:{cfg.wilcoxon_method}:{cfg.zero_method}:{correction}"


def wilcoxon_p(values: Sequence[float], cfg: ComparisonSettings) -> float | None:
    """Two-sided p of the signed-rank test, or None when it is undefined (all zero, n < 2)."""
    if len(values) < 2 or not any(values):
        return None
    result = sps.wilcoxon(
        np.asarray(values, dtype=np.float64),
        zero_method=cfg.zero_method,
        correction=cfg.correction,
        alternative="two-sided",
        method=cfg.wilcoxon_method,
    )
    p = float(result.pvalue)
    return p if math.isfinite(p) else None


def hodges_lehmann(values: Sequence[float]) -> float | None:
    """Median of the Walsh averages (x_i + x_j) / 2 over i <= j."""
    if not values:
        return None
    walsh = [
        (values[i] + values[j]) / 2.0 for i in range(len(values)) for j in range(i, len(values))
    ]
    return float(statistics.median(walsh))


def mean_interval(
    diffs: Sequence[Diff], cfg: ComparisonSettings
) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval of the mean, resampling whole ``clip_id`` units."""
    if len(diffs) < 2:
        return None, None

    def statistic(sample: Sequence[Diff]) -> Mapping[str, float]:
        return {MEAN: float(np.mean([d.value for d in sample]))}

    result = bootstrap(diffs, statistic, cfg.bootstrap_n, cfg.bootstrap_seed, cfg.ci)
    interval = result.intervals.get(MEAN)
    if interval is None or not math.isfinite(interval.low):
        return None, None
    return interval.low, interval.high


def paired_test(diffs: Sequence[Diff], cfg: ComparisonSettings, dropped: int = 0) -> TestResult:
    values = [d.value for d in diffs]
    lo, hi = mean_interval(diffs, cfg)
    return TestResult(
        n=len(values),
        dropped=dropped,
        p=wilcoxon_p(values, cfg),
        hl=hodges_lehmann(values),
        mean=float(np.mean(values)) if values else None,
        ci_lo=lo,
        ci_hi=hi,
        method=method_label(cfg),
    )


def holm(pvalues: Mapping[str, float | None]) -> dict[str, float | None]:
    """Holm-adjusted p-values; an undefined p counts in m and stays undefined."""
    m = len(pvalues)
    ordered = sorted((p, name) for name, p in pvalues.items() if p is not None)
    adjusted: dict[str, float | None] = {name: None for name in pvalues}
    running = 0.0
    for rank, (p, name) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def mde_paired(n: int, sd: float, alpha: float, power: float) -> float | None:
    """delta = (z_{1-alpha/2} + z_power) / sqrt(n) / sqrt(ARE) * sd for the signed-rank test."""
    if n < 1 or not math.isfinite(sd):
        return None
    z = float(sps.norm.ppf(1.0 - alpha / 2.0) + sps.norm.ppf(power))
    return z / math.sqrt(n) / math.sqrt(WILCOXON_ARE) * sd


def sample_sd(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None
