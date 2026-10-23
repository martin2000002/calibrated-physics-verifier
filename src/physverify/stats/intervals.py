"""Confidence intervals and exact binomial rates."""

from __future__ import annotations

from typing import Literal

from scipy.stats import binomtest

from physverify.core.models import ArtifactModel, ReportModel

RateCi = Literal["clopper_pearson", "scene_bootstrap", "clopper_pearson_scenes"]


class Interval(ReportModel):
    low: float
    high: float


class Ci(ArtifactModel):
    """Interval at the artifact's ``ci`` level; null bounds when it could not be computed."""

    low: float | None = None
    high: float | None = None
    non_finite: int | None = None
    """Bootstrap replicates whose statistic was not finite (counted, not silently dropped)."""


class Rate(ArtifactModel):
    """``k`` events out of ``n``. ``ci_method``: exact Clopper-Pearson over independent units;
    ``scene_bootstrap`` when several clips share a scene; ``clopper_pearson_scenes`` (exact over
    ``n_scenes``, conservative) when such a rate is 0 or 1 and the bootstrap would collapse."""

    value: float | None
    ci: Ci
    k: int
    n: int
    n_scenes: int | None = None
    ci_method: RateCi = "clopper_pearson"


def clopper_pearson(k: int, n: int, ci: float) -> Ci:
    if n <= 0:
        return Ci()
    interval = binomtest(k, n).proportion_ci(confidence_level=ci, method="exact")
    return Ci(low=float(interval.low), high=float(interval.high))


def clopper_pearson_lower(k: int, n: int, confidence: float) -> float:
    """One-sided exact lower bound of a binomial rate at ``confidence`` (0 when n = 0)."""
    if n <= 0:
        return 0.0
    interval = binomtest(k, n, alternative="greater").proportion_ci(
        confidence_level=confidence, method="exact"
    )
    return float(interval.low)


def rate(k: int, n: int, ci: float) -> Rate:
    return Rate(value=k / n if n > 0 else None, ci=clopper_pearson(k, n, ci), k=k, n=n)
