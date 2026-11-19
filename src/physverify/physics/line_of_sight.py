"""Per-row position covariance of a monocular lift: a depth error moves the point along its ray."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.schema.states import CoordFrame

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def line_of_sight_covariance(
    position: ArrayLike, sigma: ArrayLike, exact: ArrayLike, frame: CoordFrame, floor: float
) -> FloatArray:
    """(n, 3, 3) Sigma_t of positions lifted in CAMERA_OPENCV from per-axis ``sigma``.

    Sigma_t = r_d^2 p p^T + diag(sigma^2 - r_d^2 p^2), r_d = sigma_z / z: the relative depth
    error moves p along its ray (rank 1) and the rest stays on the diagonal, so the diagonal of
    Sigma_t is sigma^2. A rest below ``floor``^2 is raised to it (the diagonal then exceeds
    sigma^2 there), which keeps Sigma_t positive definite. Rows ``exact`` on every axis are
    ``floor``^2 I; rows without a ray (z <= 0 or a non-finite position) and every row in another
    frame keep diag(sigma^2)."""
    p = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    s2 = np.asarray(sigma, dtype=np.float64).reshape(-1, 3) ** 2
    all_exact = np.all(np.asarray(exact, dtype=bool).reshape(-1, 3), axis=1)
    diagonal = _diagonal(s2)
    if frame is not CoordFrame.CAMERA_OPENCV:
        return diagonal
    on_ray = _on_ray(p) & ~all_exact
    cov = np.where(on_ray[:, None, None], _ray_covariance(p, s2, on_ray, floor**2), diagonal)
    return np.asarray(np.where(all_exact[:, None, None], floor**2 * np.eye(3), cov))


def _diagonal(values: FloatArray) -> FloatArray:
    return np.asarray(np.einsum("nk,kl->nkl", values, np.eye(3)))


def _on_ray(p: FloatArray) -> BoolArray:
    z = p[:, 2]
    return np.asarray(np.isfinite(z) & (z > 0.0) & np.all(np.isfinite(p), axis=1))


def _ray_covariance(p: FloatArray, s2: FloatArray, on_ray: BoolArray, floor2: float) -> FloatArray:
    ray = np.where(on_ray[:, None], p, 0.0)
    z = np.where(on_ray, p[:, 2], 1.0)
    r2 = np.where(on_ray, s2[:, 2] / z**2, 0.0)
    rest = s2 - r2[:, None] * ray**2
    rest = np.where(rest < floor2, floor2, rest)
    return np.asarray(r2[:, None, None] * np.einsum("nk,nl->nkl", ray, ray) + _diagonal(rest))
