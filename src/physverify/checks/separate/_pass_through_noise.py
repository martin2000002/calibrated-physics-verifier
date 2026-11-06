"""Noise of R1_dyn's frame count and impulse: the full covariance of the change of the
least-squares rate of the descriptor D, the impulse statistic against it and the sigmas of both
terms of the magnitude n (1 - p). Pure numpy/scipy, no I/O."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ROW_COVERAGE: Final = 0.90
TAIL_DOF: Final = 3.0
DESCRIPTOR_DOF: Final = 3
_GRID: Final = 2049


def band_z() -> float:
    """z of the two-sided ``ROW_COVERAGE`` normal band: a calibrated row sigma is the band's
    half-width over this."""
    return float(stats.norm.ppf(0.5 + ROW_COVERAGE / 2.0))


def slope_weights(t: FloatArray, rows: IntArray) -> FloatArray:
    """w_t = (t - mean) / sum (t - mean)^2 over ``rows``: the least-squares slope is w @ y."""
    times = t[rows] - float(np.mean(t[rows]))
    out: FloatArray = times / float(times @ times)
    return out


def change_weights(t: FloatArray, before: IntArray, after: IntArray) -> tuple[IntArray, FloatArray]:
    """Rows of both windows and the weights c with c @ y = slope after - slope before."""
    rows = np.concatenate([before, after])
    weights = np.concatenate([-slope_weights(t, before), slope_weights(t, after)])
    return rows, weights


def lag_correlation(rows: IntArray, autocorrelation: Sequence[float]) -> FloatArray:
    """Correlation of the noise of two rows from their lag in rows: 1 at lag 0,
    ``autocorrelation[k - 1]`` at lag k, 0 beyond it."""
    table = np.concatenate([[1.0], np.asarray(autocorrelation, dtype=float), [0.0]])
    lags = np.minimum(np.abs(rows[:, None] - rows[None, :]), table.size - 1)
    out: FloatArray = table[lags]
    return out


def change_covariance(
    rows: IntArray,
    weights: FloatArray,
    descriptor: FloatArray,
    noise: tuple[FloatArray, FloatArray, FloatArray],
    row_autocorrelation: Sequence[float],
    size_autocorrelation: Sequence[float],
) -> FloatArray:
    """(3, 3) covariance of c @ D, D = ((u - c_x) / s, (v - c_y) / s, f / s), exact for the
    first-order noise dD_t = (du_t / s_t, dv_t / s_t, 0) - D_t e_t: ``noise`` = (s, pixel sigma
    of u and v, relative sigma of s) per row; u, v and e independent, each correlated across
    rows by its own autocorrelation. Cov = (a' R_r a) I_xy + B' R_s B, a_t = c_t sigma_t / s_t,
    B_t = c_t sigma_e,t D_t."""
    size, pixel, relative = noise
    a = weights * pixel[rows] / size[rows]
    b = (weights * relative[rows])[:, None] * descriptor[rows]
    out: FloatArray = b.T @ lag_correlation(rows, size_autocorrelation) @ b
    out[:2, :2] += float(a @ lag_correlation(rows, row_autocorrelation) @ a) * np.eye(2)
    return out


def positive_definite(covariance: FloatArray) -> bool:
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        return False
    return True


def impulse_statistic(change: FloatArray, covariance: FloatArray) -> float:
    """q = change' covariance^-1 change: chi-square with 3 dof when the rate does not change."""
    return float(change @ np.linalg.solve(covariance, change))


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
