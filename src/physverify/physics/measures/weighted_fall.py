"""R5a whitened: parabolic fall fit weighted by the per-row position sigma, and the Wald contrast
between two fitted accelerations [accel_wald_chi2_3]."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vec = NDArray[np.float64]
MIN_ROWS = 3


@dataclass(frozen=True, eq=False)
class WeightedFallFit:
    """Per axis, the acceleration of p(t) = p0 + v0 tau + a tau^2 / 2 (tau = t - t[0]) fitted with
    weights 1 / sigma^2, and its variance [(X^T W X)^-1]_aa under that sigma."""

    a: Vec
    a_var: Vec


def weighted_fall_fit(
    t: ArrayLike, positions: ArrayLike, sigma: ArrayLike
) -> WeightedFallFit | None:
    """None when an axis has fewer than three rows with a finite position and a sigma > 0, or a
    singular normal matrix."""
    times = np.asarray(t, dtype=np.float64).reshape(-1)
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    s = np.asarray(sigma, dtype=np.float64).reshape(-1, 3)
    tau = times - times[0]
    design = np.column_stack([np.ones_like(tau), tau, 0.5 * tau**2])
    a, a_var = np.full(3, math.nan), np.full(3, math.nan)
    for axis in range(3):
        fitted = _axis(design, p[:, axis], s[:, axis])
        if fitted is None:
            return None
        a[axis], a_var[axis] = fitted
    return WeightedFallFit(a=a, a_var=a_var)


def _axis(design: Vec, y: Vec, s: Vec) -> tuple[float, float] | None:
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0.0)
    if int(ok.sum()) < MIN_ROWS:
        return None
    w = 1.0 / s[ok] ** 2
    x = design[ok]
    try:
        inverse = np.linalg.inv(x.T @ (w[:, None] * x))
    except np.linalg.LinAlgError:
        return None
    coef = inverse @ (x.T @ (w * y[ok]))
    return float(coef[2]), float(inverse[2, 2])


def wald_terms(first: WeightedFallFit, second: WeightedFallFit) -> Vec:
    """(a1_d - a2_d)^2 / (var1_d + var2_d) per axis of two independent fits."""
    return np.asarray((first.a - second.a) ** 2 / (first.a_var + second.a_var), dtype=np.float64)


def accel_wald(first: WeightedFallFit, second: WeightedFallFit) -> float:
    """Sum of ``wald_terms``: chi-square with three degrees of freedom when both fits share one
    acceleration and sigma is the noise's."""
    return float(np.sum(wald_terms(first, second)))
