"""Agreement statistics (ADR 0011 D6, D8). Labels are 0/1 with 1 = implausible (positive)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from physverify.stats.bootstrap import BootstrapResult, bootstrap

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


@dataclass(frozen=True)
class CaptionMcNemar:
    """McNemar by caption sign flip: ``statistic`` is the sum over captions of the per-caption
    difference (verifier-only minus VLM-only correct clips); ``method`` exact or monte_carlo."""

    statistic: int
    discordant_captions: int
    p_value: float
    method: str
    replicas: int


EXACT: Final = "exact"
MONTE_CARLO: Final = "monte_carlo"
_TIE: Final = 1e-9


def caption_differences(y: Labels, a: Labels, b: Labels, captions: Sequence[str]) -> list[int]:
    """Per caption, clips only ``a`` gets right minus clips only ``b`` gets right."""
    sums: dict[str, int] = {}
    diff = ((a == y) & (b != y)).astype(int) - ((a != y) & (b == y)).astype(int)
    for caption, d in zip(captions, diff.tolist(), strict=True):
        sums[caption] = sums.get(caption, 0) + int(d)
    return [d for d in sums.values() if d != 0]


def _sign_flip_exact(diffs: Sequence[int], observed: int) -> float:
    """P(|sum s_c d_c| >= |observed|), s_c = +-1 with probability 1/2 (exact convolution)."""
    dist: dict[int, float] = {0: 1.0}
    for d in diffs:
        nxt: dict[int, float] = {}
        for total, prob in dist.items():
            for step in (d, -d):
                nxt[total + step] = nxt.get(total + step, 0.0) + prob / 2
        dist = nxt
    return float(min(1.0, sum(p for t, p in dist.items() if abs(t) >= abs(observed) - _TIE)))


def _sign_flip_monte_carlo(diffs: Sequence[int], observed: int, n: int, seed: int) -> float:
    """(1 + hits) / (n + 1) over ``n`` random sign flips (Phipson and Smyth 2010)."""
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1, 1]), size=(n, len(diffs)))
    totals = np.abs(signs @ np.asarray(diffs, dtype=np.int64))
    hits = int(np.sum(totals >= abs(observed) - _TIE))
    return (1 + hits) / (n + 1)


def clustered_mcnemar(
    y: Labels,
    a: Labels,
    b: Labels,
    captions: Sequence[str],
    *,
    exact_max: int,
    n: int,
    seed: int,
) -> CaptionMcNemar:
    """Two-sided McNemar that keeps the videos of one caption together: under H0 each caption's
    difference is equally likely to have either sign. Exact up to ``exact_max`` discordant
    captions (then it equals ``mcnemar_exact`` with one caption per clip), Monte Carlo above;
    NaN when Monte Carlo is needed with ``n`` = 0."""
    diffs = caption_differences(y, a, b, captions)
    observed = sum(diffs)
    m = len(diffs)
    if m <= exact_max:
        return CaptionMcNemar(observed, m, _sign_flip_exact(diffs, observed), EXACT, 0)
    if n == 0:
        return CaptionMcNemar(observed, m, math.nan, MONTE_CARLO, 0)
    p_value = _sign_flip_monte_carlo(diffs, observed, n, seed)
    return CaptionMcNemar(observed, m, p_value, MONTE_CARLO, n)


def mde_accuracy(n: float, discordant: float, alpha: float, power: float) -> float:
    """Smallest accuracy difference a paired McNemar test detects (Connor 1987), by bisection on
    n * delta^2 = (z_a sqrt(d) + z_b sqrt(d - delta^2))^2 over 0 < delta <= d (an accuracy
    difference cannot exceed the discordant proportion d). NaN without discordant pairs or when
    even delta = d is not detectable with ``n`` pairs. ``n`` may be an effective size
    (pairs over a design effect)."""
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
class DesignEffect:
    """Estimated design effect of clustering ``indicator`` by caption (Kish): ``deff`` = 1 +
    (m_w - 1) icc, with m_w = sum(n_i^2) / N the size-weighted mean cluster size (it equals the
    mean when every cluster has the same size). ``icc`` is the one-way ANOVA estimator, clipped
    to [0, 1], whose mean cluster size for unequal sizes is m0 = (N - sum(n_i^2) / N) / (k - 1)."""

    deff: float
    icc: float
    m0: float
    clusters: int
    n: int


def design_effect(clusters: Sequence[str], indicator: Sequence[float] | Labels) -> DesignEffect:
    """NaN without two clusters; every cluster of one item gives deff 1 (no clustering)."""
    groups: dict[str, list[float]] = {}
    for cluster, value in zip(clusters, indicator, strict=True):
        groups.setdefault(cluster, []).append(float(value))
    n, k = len(clusters), len(groups)
    if k < 2:
        return DesignEffect(math.nan, math.nan, math.nan, k, n)
    sizes = np.asarray([len(v) for v in groups.values()], dtype=np.float64)
    weighted = float(np.sum(sizes**2) / n)
    m0 = (n - weighted) / (k - 1)
    icc = _anova_icc(list(groups.values()), m0) if n > k else 0.0
    return DesignEffect(1.0 + (weighted - 1.0) * icc, icc, m0, k, n)


def _anova_icc(groups: Sequence[Sequence[float]], m0: float) -> float:
    values = np.concatenate([np.asarray(g, dtype=np.float64) for g in groups])
    grand = float(np.mean(values))
    n, k = values.size, len(groups)
    between = sum(len(g) * (float(np.mean(g)) - grand) ** 2 for g in groups) / (k - 1)
    within = sum(float(np.sum((np.asarray(g) - np.mean(g)) ** 2)) for g in groups) / (n - k)
    denominator = between + (m0 - 1.0) * within
    if denominator <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, (between - within) / denominator)))


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
