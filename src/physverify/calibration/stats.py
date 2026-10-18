"""Small statistics shared by the calibration steps (ADR 0009 D5-D11)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import binomtest, chi2, mannwhitneyu

from physverify.calibration.schema import Ci, Rate
from physverify.diagnosis.factorial import bootstrap

FloatArray = NDArray[np.float64]
MAD_TO_SIGMA = 0.6745
"""median(|z|) of a standard normal: median(|z|) / 0.6745 is a robust scale."""
NEWTON_STEPS = 100
NEWTON_TOL = 1e-10


def clopper_pearson(k: int, n: int, ci: float) -> Ci:
    if n <= 0:
        return Ci()
    interval = binomtest(k, n).proportion_ci(confidence_level=ci, method="exact")
    return Ci(low=float(interval.low), high=float(interval.high))


def rate(k: int, n: int, ci: float) -> Rate:
    return Rate(value=k / n if n > 0 else None, ci=clopper_pearson(k, n, ci), k=k, n=n)


def conformal_index(n: int, alpha: float) -> int:
    """1-based rank k = ceil((n + 1)(1 - alpha)); k > n means the guarantee needs more data."""
    return math.ceil((n + 1) * (1.0 - alpha) - 1e-12)


@dataclass(frozen=True)
class Conformal:
    value: float | None
    sufficient: bool


def conformal_quantile(values: ArrayLike, alpha: float) -> Conformal:
    """m_(k) of the exchangeable samples; the maximum (and insufficient) when k > n."""
    ordered = np.sort(_finite(values))
    if not ordered.size:
        return Conformal(None, False)
    k = conformal_index(int(ordered.size), alpha)
    if k > ordered.size:
        return Conformal(float(ordered[-1]), False)
    return Conformal(float(ordered[k - 1]), True)


def grouped_folds(groups: Sequence[str], k: int, seed: int) -> tuple[frozenset[str], ...]:
    """At most ``k`` disjoint sets of whole groups (scenes), shuffled by ``seed``."""
    unique = sorted(set(groups))
    if not unique:
        return ()
    order = np.random.default_rng(seed).permutation(len(unique))
    n_folds = min(k, len(unique))
    folds: list[set[str]] = [set() for _ in range(n_folds)]
    for position, index in enumerate(order):
        folds[position % n_folds].add(unique[int(index)])
    return tuple(frozenset(fold) for fold in folds)


@dataclass(frozen=True)
class SceneBlock[T]:
    """Adapter for ``diagnosis.factorial.bootstrap``: its ``clip_id`` is the scene."""

    clip_id: str
    item: T


def scene_bootstrap[T](
    blocks: Sequence[SceneBlock[T]],
    statistic: Callable[[Sequence[T]], Mapping[str, float]],
    n: int,
    seed: int,
    ci: float,
) -> dict[str, Ci]:
    """Percentile intervals from resampling whole scenes (a drawn twice scene counts twice);
    ``non_finite`` counts the replicates whose statistic was undefined."""
    result = bootstrap(blocks, lambda sample: statistic([b.item for b in sample]), n, seed, ci)
    return {
        key: Ci(
            low=_or_none(i.low),
            high=_or_none(i.high),
            non_finite=result.non_finite.get(key, 0),
        )
        for key, i in result.intervals.items()
    }


RATE_KEY = "rate"


def clustered_rate(flags: Sequence[tuple[str, bool]], n_boot: int, seed: int, ci: float) -> Rate:
    """Rate of (scene, event) flags. One flag per scene: Clopper-Pearson. Otherwise the
    scene-bootstrap percentile interval, or Clopper-Pearson over the scenes when the rate is 0 or
    1 (the bootstrap would collapse to a point)."""
    k, n = sum(f for _, f in flags), len(flags)
    scenes = len({scene for scene, _ in flags})
    base = rate(k, n, ci)
    if scenes == n:
        return base.model_copy(update={"n_scenes": scenes})
    if k in (0, n):
        bound = clopper_pearson(0 if k == 0 else scenes, scenes, ci)
        return base.model_copy(
            update={"ci": bound, "n_scenes": scenes, "ci_method": "clopper_pearson_scenes"}
        )
    blocks = [SceneBlock(scene, flag) for scene, flag in flags]
    interval = scene_bootstrap(
        blocks, lambda s: {RATE_KEY: float(np.mean(s))}, n_boot, seed, ci
    ).get(RATE_KEY, Ci())
    return base.model_copy(
        update={"ci": interval, "n_scenes": scenes, "ci_method": "scene_bootstrap"}
    )


def auroc(positives: ArrayLike, negatives: ArrayLike) -> float:
    """P(m_pos > m_neg) + P(tie) / 2 (Mann-Whitney U / n_pos n_neg); NaN if a side is empty."""
    pos, neg = _finite(positives), _finite(negatives)
    if not pos.size or not neg.size:
        return math.nan
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u) / (pos.size * neg.size)


def ece(p: ArrayLike, y: ArrayLike, weights: ArrayLike | None = None, bins: int = 10) -> float:
    """Expected calibration error over equal-width bins of p (weighted)."""
    probs, labels = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = np.ones_like(probs) if weights is None else np.asarray(weights, dtype=np.float64)
    if not probs.size or w.sum() <= 0:
        return math.nan
    index = np.minimum((probs * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            weight = w[mask].sum()
            gap = abs(np.dot(w[mask], probs[mask]) - np.dot(w[mask], labels[mask])) / weight
            total += weight * gap
    return float(total / w.sum())


@dataclass(frozen=True)
class Logistic:
    """P(y = 1 | x) = expit(a + b x)."""

    a: float
    b: float

    def predict(self, x: ArrayLike) -> FloatArray:
        return np.asarray(expit(self.a + self.b * np.asarray(x, dtype=np.float64)))


def fit_logistic(
    x: ArrayLike,
    y: ArrayLike,
    *,
    l2: float,
    weights: ArrayLike | None = None,
    monotone: bool = False,
) -> Logistic | None:
    """Weighted logistic regression with a fixed total L2 penalty (a weak prior, so the fit is
    finite under separation and unbiased for large n); ``monotone`` forces b >= 0. None without
    both classes."""
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = np.ones_like(xs) if weights is None else np.asarray(weights, dtype=np.float64)
    if not xs.size or ys.min() == ys.max():
        return None
    centre = float(np.average(xs, weights=w))
    a, b = _newton(xs - centre, ys, w, l2, fit_slope=True)
    if monotone and b < 0.0:
        a, b = _newton(xs - centre, ys, w, l2, fit_slope=False)
    return Logistic(a=a - b * centre, b=b)


def _newton(
    x: FloatArray, y: FloatArray, w: FloatArray, l2: float, fit_slope: bool
) -> tuple[float, float]:
    design = np.column_stack([np.ones_like(x), x]) if fit_slope else np.ones((x.size, 1))
    beta = np.zeros(design.shape[1])
    penalty = l2 * np.eye(design.shape[1])
    for _ in range(NEWTON_STEPS):
        p = expit(design @ beta)
        gradient = design.T @ (w * (y - p)) - penalty @ beta
        hessian = design.T @ (design * (w * p * (1.0 - p))[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if float(np.max(np.abs(step))) < NEWTON_TOL:
            break
    return float(beta[0]), float(beta[1]) if fit_slope else 0.0


@dataclass(frozen=True)
class ProfileInterval:
    """Interval of x0 in x space; None on a side that does not close inside the range."""

    low: float | None
    high: float | None


PROFILE_GRID = 241


def profile_loglik(x: FloatArray, y: FloatArray, x0: float, target: float, l2: float) -> float:
    """max over b >= 0 of the log-likelihood of expit(target + b (x - x0)) (penalty l2 b^2 / 2):
    the logistic re-parametrised so that P(y = 1 | x0) = expit(target)."""
    z = x - x0
    b = 1.0
    for _ in range(NEWTON_STEPS):
        p = expit(target + b * z)
        gradient = float(z @ (y - p)) - l2 * b
        curvature = float((z * z) @ (p * (1.0 - p))) + l2
        if curvature <= 0.0:
            break
        new = max(b + gradient / curvature, 0.0)
        done = abs(new - b) < NEWTON_TOL
        b = new
        if done:
            break
    eta = target + b * z
    return float(y @ eta - np.logaddexp(0.0, eta).sum() - 0.5 * l2 * b * b)


def profile_interval(
    x: ArrayLike, y: ArrayLike, target: float, l2: float, ci: float, bounds: tuple[float, float]
) -> ProfileInterval:
    """{x0 : 2 (l_max - l(x0)) <= chi2_1(ci)} over ``bounds`` (Wilks), scanned on a grid and
    refined by bisection."""
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    grid = np.linspace(bounds[0], bounds[1], PROFILE_GRID)
    values = np.array([profile_loglik(xs, ys, float(g), target, l2) for g in grid])
    cut = float(values.max()) - float(chi2.ppf(ci, 1)) / 2.0

    def excess(g: float) -> float:
        return profile_loglik(xs, ys, g, target, l2) - cut

    best = int(values.argmax())
    low = best
    while low > 0 and values[low - 1] >= cut:
        low -= 1
    high = best
    while high < grid.size - 1 and values[high + 1] >= cut:
        high += 1
    return ProfileInterval(
        low=None if low == 0 else float(brentq(excess, grid[low - 1], grid[low])),
        high=None if high == grid.size - 1 else float(brentq(excess, grid[high], grid[high + 1])),
    )


def balanced_weights(y: ArrayLike) -> FloatArray:
    """Equal total weight for each class (the design prevalence must not move p)."""
    labels = np.asarray(y, dtype=np.float64)
    n_pos = float(labels.sum())
    n_neg = float(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.ones_like(labels)
    return np.where(labels > 0.5, labels.size / (2.0 * n_pos), labels.size / (2.0 * n_neg))


def mad(values: ArrayLike) -> float:
    """Median absolute deviation scaled to sigma; 1 when it is 0 or undefined."""
    finite = _finite(values)
    if not finite.size:
        return 1.0
    spread = float(np.median(np.abs(finite - np.median(finite)))) / MAD_TO_SIGMA
    return spread if spread > 0.0 else 1.0


def median_or_none(values: ArrayLike) -> float | None:
    finite = _finite(values)
    return float(np.median(finite)) if finite.size else None


def _finite(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64).ravel()
    return array[np.isfinite(array)]


def _or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None
