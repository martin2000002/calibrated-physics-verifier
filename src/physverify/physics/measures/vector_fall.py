"""Constant-acceleration fall fitted by generalized least squares on a per-row 3x3 position
covariance, and the tilt of its acceleration away from gravity [accel_tilt_chi2_4]."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.physics.measures.weighted_fall import MIN_ROWS

FloatArray = NDArray[np.float64]
ACCEL = slice(6, 9)


@dataclass(frozen=True, eq=False)
class VectorFallFit:
    """The fitted acceleration and its 3x3 covariance under the row covariances."""

    a: FloatArray
    a_cov: FloatArray


@dataclass(frozen=True)
class BallisticFallFit:
    """g along the given gravity direction (table units / s^2) and its variance."""

    g: float
    g_var: float


def gls_fall_fit(t: ArrayLike, position: ArrayLike, cov: ArrayLike) -> VectorFallFit | None:
    """p_t = p0 + v0 tau + a tau^2 / 2 with p0, v0 and a free (nine unknowns); None below three
    distinct times with a finite position and covariance, or with a singular system."""
    found = _fit(t, position, cov, None)
    if found is None:
        return None
    coef, inverse = found
    return VectorFallFit(np.asarray(coef[ACCEL]), np.asarray(inverse[ACCEL, ACCEL]))


def gls_ballistic_fit(
    t: ArrayLike, position: ArrayLike, cov: ArrayLike, g_dir: ArrayLike
) -> BallisticFallFit | None:
    """p_t = p0 + v0 tau + g g_dir tau^2 / 2 with p0, v0 and the scalar g free (seven unknowns);
    None as ``gls_fall_fit``."""
    found = _fit(t, position, cov, np.asarray(g_dir, dtype=np.float64).reshape(3))
    if found is None:
        return None
    coef, inverse = found
    return BallisticFallFit(float(coef[6]), float(inverse[6, 6]))


def tilt_statistic(fit: VectorFallFit, g_dir: ArrayLike, g_dir_sigma: float) -> float:
    """a_h^T (E S_a E^T + |a|^2 sigma_g^2 I_2)^-1 a_h with a_h = E a, E a (2, 3) orthonormal
    basis of the plane orthogonal to ``g_dir``: the acceleration's departure from gravity against
    its own covariance and a gravity tilt of ``g_dir_sigma`` per axis (chi-square with two degrees
    of freedom when a is parallel to gravity). NaN when that matrix is singular."""
    e = _orthogonal_basis(g_dir)
    a2 = float(fit.a @ fit.a)
    return _mahalanobis(e @ fit.a, e @ fit.a_cov @ e.T + a2 * g_dir_sigma**2 * np.eye(2))


def _orthogonal_basis(normal: ArrayLike) -> FloatArray:
    # The statistic does not depend on the basis, but its floating bits do: on this statistic
    # ``geometry.plane_basis`` drifts by ~1e-13 relative, ~0.02 absolute near magnitudes of 1e12.
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    _, _, vt = np.linalg.svd((n / np.linalg.norm(n))[None, :])
    return np.asarray(vt[1:3])


def _fit(
    t: ArrayLike, position: ArrayLike, cov: ArrayLike, g_dir: FloatArray | None
) -> tuple[FloatArray, FloatArray] | None:
    rows = _rows(t, position, cov)
    return None if rows is None else _gls(*rows, g_dir)


def _rows(
    t: ArrayLike, position: ArrayLike, cov: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray] | None:
    """Centred times, positions and covariances of the rows where all three are finite."""
    times = np.asarray(t, dtype=np.float64).reshape(-1)
    p = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(cov, dtype=np.float64).reshape(-1, 3, 3)
    keep = np.isfinite(times) & np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(c), axis=(1, 2))
    if np.unique(times[keep]).size < MIN_ROWS:
        return None
    return times[keep] - times[keep].mean(), p[keep], c[keep]


def _design(tau: FloatArray, g_dir: FloatArray | None) -> FloatArray:
    """(n, 3, k) blocks [I, tau I, tau^2 / 2 I] (k = 9) or [I, tau I, tau^2 / 2 g_dir] (k = 7)."""
    n, eye = tau.size, np.eye(3)
    half = 0.5 * tau**2
    last = half[:, None, None] * eye if g_dir is None else (half[:, None] * g_dir)[:, :, None]
    return np.concatenate([np.broadcast_to(eye, (n, 3, 3)), tau[:, None, None] * eye, last], axis=2)


def _gls(
    tau: FloatArray, p: FloatArray, cov: FloatArray, g_dir: FloatArray | None
) -> tuple[FloatArray, FloatArray] | None:
    """Coefficients (X^T W X)^-1 X^T W p with W = Sigma_t^-1 per row, and (X^T W X)^-1."""
    x = _design(tau, g_dir)
    try:
        w = np.linalg.inv(cov)
        normal = np.einsum("nki,nkl,nlj->ij", x, w, x)
        inverse = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None
    weighted = np.einsum("nki,nkl->inl", x, w).reshape(x.shape[2], -1)
    coef = np.linalg.solve(normal, weighted) @ p.reshape(-1)
    return np.asarray(coef), np.asarray(inverse)


def _mahalanobis(d: FloatArray, s: FloatArray) -> float:
    try:
        return float(d @ np.linalg.solve(s, d))
    except np.linalg.LinAlgError:
        return math.nan
