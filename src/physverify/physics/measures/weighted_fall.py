"""R5a whitened: parabolic fall fit weighted by the per-row position sigma, and the Wald contrast
between two fitted accelerations [accel_wald_chi2_3]."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.physics.tracks import poly_fit

Vec = NDArray[np.float64]
MIN_ROWS = 3


@dataclass(frozen=True, eq=False)
class WeightedFallFit:
    """Per axis, the acceleration of the degree-2 ``poly_fit`` weighted by 1 / sigma^2 and its
    variance under that sigma."""

    a: Vec
    a_var: Vec


def weighted_fall_fit(
    t: ArrayLike, positions: ArrayLike, sigma: ArrayLike
) -> WeightedFallFit | None:
    """None when fewer than three distinct times have a finite position and a sigma > 0 on every
    axis, or the normal matrix is singular."""
    times = np.asarray(t, dtype=np.float64).reshape(-1)
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    s = np.asarray(sigma, dtype=np.float64).reshape(-1, 3)
    keep = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(s) & (s > 0.0), axis=1)
    if np.unique(times[keep]).size < MIN_ROWS:
        return None
    try:
        fit = poly_fit(times[keep], p[keep], s[keep], 2)
    except np.linalg.LinAlgError:
        return None
    if fit.cov is None:
        return None
    return WeightedFallFit(
        a=np.asarray(2.0 * fit.coef[2], dtype=np.float64),
        a_var=np.asarray(4.0 * fit.cov[:, 2, 2], dtype=np.float64),
    )


def wald_terms(first: WeightedFallFit, second: WeightedFallFit) -> Vec:
    """(a1_d - a2_d)^2 / (var1_d + var2_d) per axis of two independent fits."""
    return np.asarray((first.a - second.a) ** 2 / (first.a_var + second.a_var), dtype=np.float64)


def accel_wald(first: WeightedFallFit, second: WeightedFallFit) -> float:
    """Sum of ``wald_terms``: chi-square with three degrees of freedom per candidate when both
    fits share one acceleration and sigma is the noise's, white and independent per axis."""
    return float(np.sum(wald_terms(first, second)))
