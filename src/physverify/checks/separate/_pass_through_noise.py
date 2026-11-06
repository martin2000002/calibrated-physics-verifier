"""Noise of R1_dyn's frame count and impulse: per-row variance of the descriptor D, variance of
its least-squares rate, the impulse statistic against that noise and the sigmas of both terms
of the magnitude n (1 - p). Pure numpy/scipy, no I/O."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]

ROW_COVERAGE: Final = 0.90
TAIL_DOF: Final = 3.0
DESCRIPTOR_DOF: Final = 3
_GRID: Final = 2049


def band_z() -> float:
    """z of the two-sided ``ROW_COVERAGE`` normal band: a calibrated row sigma is the band's
    half-width over this."""
    return float(stats.norm.ppf(0.5 + ROW_COVERAGE / 2.0))


def descriptor_variance(
    descriptor: FloatArray, size: FloatArray, focal: float, row: FloatArray, size_rel: FloatArray
) -> FloatArray:
    """(n, 3) variance of D = ((u - c_x) / s, (v - c_y) / s, f / s) per row: the pixel sigma
    ``row`` on the column and the row, the relative size sigma ``size_rel`` on s,
    var(D_xy) = (row / s)^2 + (D_xy size_rel)^2 and var(D_z) = (f / s size_rel)^2."""
    pixel = (row / size)[:, None] ** 2 + (descriptor[:, :2] * size_rel[:, None]) ** 2
    depth = (focal / size * size_rel) ** 2
    return np.column_stack([pixel, depth])


def slope_variance(
    t: FloatArray, variance: FloatArray, rows: NDArray[np.int64], autocorrelation: float
) -> FloatArray:
    """(3,) variance of the least-squares slope over ``rows``: sum_t w_t^2 var_t, w_t =
    (t - mean) / sum (t - mean)^2, times the autocorrelation factor rho = 1 + 2 sum_k rho_k."""
    times = t[rows] - float(np.mean(t[rows]))
    weights = times / float(times @ times)
    out: FloatArray = (weights**2) @ variance[rows] * autocorrelation
    return out


def impulse_statistic(change: FloatArray, variance: FloatArray) -> float:
    """q = sum_i change_i^2 / variance_i: chi-square with 3 dof when the rate does not change."""
    return float(np.sum(change**2 / variance))


def evidence(q: float) -> float:
    """F_chi2_3(q): uniform on [0, 1] without an impulse, near 1 with a clear one."""
    return float(stats.chi2.cdf(q, DESCRIPTOR_DOF))


def evidence_sigma(statistics: list[float]) -> float:
    """Standard deviation of max_b F(Q_b), Q_b ~ noncentral chi2_3 with the plug-in
    noncentrality max(q_b - 3, 0), from the CDF of the maximum (product of the bodies' CDFs)."""
    x = np.linspace(0.0, 1.0, _GRID)
    quantiles = stats.chi2.ppf(x, DESCRIPTOR_DOF)
    joint = np.ones_like(x)
    for q in statistics:
        noncentrality = max(q - DESCRIPTOR_DOF, 0.0)
        joint *= (
            x if noncentrality == 0.0 else stats.ncx2.cdf(quantiles, DESCRIPTOR_DOF, noncentrality)
        )
    survival = 1.0 - joint
    mean = float(np.trapezoid(survival, x))
    second = float(np.trapezoid(2.0 * x * survival, x))
    return math.sqrt(max(second - mean**2, 0.0))


def flip_probability(margin: FloatArray, sigma: FloatArray, tail_dof: float | None) -> FloatArray:
    """Probability that a row's match decision flips: F(-|margin| / sigma), F normal or a
    Student t with ``tail_dof`` rescaled to the same ``ROW_COVERAGE`` quantile; 0 where sigma
    is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        standard = np.where(sigma > 0.0, -np.abs(margin) / sigma, -np.inf)
    if tail_dof is None:
        out: FloatArray = stats.norm.cdf(standard)
        return out
    stretch = float(stats.t.ppf(0.5 + ROW_COVERAGE / 2.0, tail_dof)) / band_z()
    widened: FloatArray = stats.t.cdf(standard * stretch, tail_dof)
    return widened


def count_sigma(flip: FloatArray, frames_per_row: int) -> float:
    """sigma_n = sqrt(sum_t pi_t (1 - pi_t)) times the native frames per row [frames]."""
    return float(frames_per_row * math.sqrt(float(np.sum(flip * (1.0 - flip)))))


def magnitude_sigma(
    frames: int, impulse: float, frames_sigma: float, impulse_sigma: float
) -> float:
    """sigma_m of m = n (1 - p): sqrt((1 - p)^2 sigma_n^2 + n^2 sigma_p^2) [frames]."""
    return math.hypot((1.0 - impulse) * frames_sigma, frames * impulse_sigma)
