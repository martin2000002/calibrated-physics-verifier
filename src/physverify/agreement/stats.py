"""Agreement statistics (ADR 0011 D6, D8). Labels are 0/1 with 1 = implausible (positive)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.diagnosis.factorial import BootstrapResult, bootstrap

Labels = NDArray[np.int_]
Scores = NDArray[np.float64]


def as_labels(values: Sequence[bool]) -> Labels:
    return np.asarray(values, dtype=np.int_)


def accuracy(y: Labels, p: Labels) -> float:
    return float(np.mean(y == p)) if y.size else math.nan


def _rate(y: Labels, p: Labels, label: int) -> float:
    mask = y == label
    return float(np.mean(p[mask] == label)) if mask.any() else math.nan


def balanced_accuracy(y: Labels, p: Labels) -> float:
    """Mean of sensitivity and specificity; NaN when a class is absent."""
    return (_rate(y, p, 1) + _rate(y, p, 0)) / 2.0


def f1_positive(y: Labels, p: Labels) -> float:
    tp = int(np.sum((y == 1) & (p == 1)))
    wrong = int(np.sum(y != p))
    return 2.0 * tp / (2.0 * tp + wrong) if tp + wrong else math.nan


def cohen_kappa(y: Labels, p: Labels) -> float:
    """(p_o - p_e) / (1 - p_e) with p_e from the marginals; NaN when p_e = 1."""
    if not y.size:
        return math.nan
    observed = float(np.mean(y == p))
    expected = float(np.mean(y) * np.mean(p) + (1 - np.mean(y)) * (1 - np.mean(p)))
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else math.nan


def auc(y: Labels, score: Scores) -> float:
    """Mann-Whitney AUC of ``score`` for the positive class, ties count 1/2, NaN scores dropped."""
    keep = np.isfinite(score)
    positives, negatives = score[keep & (y == 1)], score[keep & (y == 0)]
    if not positives.size or not negatives.size:
        return math.nan
    greater = np.sum(positives[:, None] > negatives[None, :])
    ties = np.sum(positives[:, None] == negatives[None, :])
    return float((greater + 0.5 * ties) / (positives.size * negatives.size))


def correlations(score: Scores, human: Scores) -> tuple[float, float]:
    """(Pearson, Spearman) over pairs where both are finite."""
    from scipy import stats

    keep = np.isfinite(score) & np.isfinite(human)
    a, b = score[keep], human[keep]
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return math.nan, math.nan
    return float(stats.pearsonr(a, b).statistic), float(stats.spearmanr(a, b).statistic)


def classification(y: Labels, p: Labels) -> dict[str, float]:
    return {
        "accuracy": accuracy(y, p),
        "balanced_accuracy": balanced_accuracy(y, p),
        "f1_implausible": f1_positive(y, p),
        "kappa": cohen_kappa(y, p),
    }


@dataclass(frozen=True)
class McNemar:
    """b: only A right; c: only B right; exact two-sided binomial p (1 when b + c = 0)."""

    b: int
    c: int
    p_value: float


def mcnemar_exact(y: Labels, a: Labels, b: Labels) -> McNemar:
    from scipy.stats import binomtest

    only_a = int(np.sum((a == y) & (b != y)))
    only_b = int(np.sum((a != y) & (b == y)))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5, alternative="two-sided").pvalue)
    return McNemar(only_a, only_b, p)


def mde_accuracy(n: int, discordant: float, alpha: float, power: float) -> float:
    """Smallest accuracy difference a paired McNemar test detects (Connor 1987), by bisection on
    n * delta^2 = (z_a sqrt(d) + z_b sqrt(d - delta^2))^2 over 0 < delta <= d (an accuracy
    difference cannot exceed the discordant proportion d). NaN without discordant pairs or when
    even delta = d is not detectable with ``n`` pairs."""
    from scipy.stats import norm

    if n <= 0 or not 0.0 < discordant <= 1.0:
        return math.nan
    z_a, z_b = float(norm.ppf(1 - alpha / 2)), float(norm.ppf(power))

    def needed(delta: float) -> float:
        return (
            z_a * math.sqrt(discordant) + z_b * math.sqrt(discordant - delta**2)
        ) ** 2 / delta**2

    low, high = 1e-9, discordant
    if needed(high) > n:
        return math.nan
    for _ in range(100):
        mid = (low + high) / 2.0
        low, high = (mid, high) if needed(mid) > n else (low, mid)
    return high


@dataclass(frozen=True)
class CaptionBlock[T]:
    """Bootstrap adapter: ``factorial.bootstrap`` resamples by ``clip_id``, here the caption."""

    item: T
    caption: str

    @property
    def clip_id(self) -> str:
        return self.caption


def caption_bootstrap[T](
    items: Sequence[T],
    caption: Callable[[T], str],
    statistic: Callable[[Sequence[T]], Mapping[str, float]],
    n: int,
    seed: int,
    ci: float,
) -> BootstrapResult:
    """Percentile intervals resampling whole captions (videos of one prompt are correlated)."""
    blocks = [CaptionBlock(item, caption(item)) for item in items]
    return bootstrap(blocks, lambda sample: statistic([b.item for b in sample]), n, seed, ci)
