"""The floor's normal from a depth map: a RANSAC plane through back-projected points of the lower
image outside every (dilated) object mask, oriented away from the camera (downward)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation

from physverify.extraction.pinhole import ray_points

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class FloorSampling:
    """Fixed a priori, never fitted on calibration clips."""

    frames: int = 8
    points_per_frame: int = 3000
    mask_dilate_px: int = 5
    first_row_frac: float = 0.6
    ransac_iters: int = 1000
    inlier_rel: float = 0.01


def floor_points(
    z: FloatArray,
    valid: BoolArray,
    masks: BoolArray,
    intrinsics: FloatArray,
    rng: np.random.Generator,
    sampling: FloorSampling,
) -> FloatArray:
    """Camera-frame points X = z K^-1 [u, v, 1] of floor candidates on evenly spaced frames."""
    n_t, height, _ = z.shape
    cut = int(sampling.first_row_frac * height)
    found = []
    for t in np.unique(np.linspace(0, n_t - 1, min(n_t, sampling.frames)).round().astype(int)):
        objects = binary_dilation(masks[t].any(axis=0), iterations=sampling.mask_dilate_px)
        ok = valid[t] & np.isfinite(z[t]) & (z[t] > 0) & ~objects
        ok[:cut] = False
        rows, cols = np.nonzero(ok)
        if rows.size == 0:
            continue
        pick = rng.choice(rows.size, min(sampling.points_per_frame, rows.size), replace=False)
        rows, cols = rows[pick], cols[pick]
        depth = z[t][rows, cols]
        found.append(ray_points(cols + 0.5, rows + 0.5, depth, intrinsics))
    return np.concatenate(found) if found else np.empty((0, 3))


def plane_normal(
    points: FloatArray, rng: np.random.Generator, sampling: FloorSampling
) -> FloatArray | None:
    """Unit normal n of the largest consensus plane, with n·centroid > 0; None below 3 points."""
    if points.shape[0] < 3:
        return None
    threshold = sampling.inlier_rel * float(np.median(np.linalg.norm(points, axis=1)))
    best, best_count = None, 2
    for _ in range(sampling.ransac_iters):
        a, b, c = points[rng.choice(points.shape[0], 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            continue
        inliers = np.abs((points - a) @ (normal / norm)) < threshold
        if int(inliers.sum()) > best_count:
            best, best_count = inliers, int(inliers.sum())
    if best is None:
        return None
    chosen = points[best]
    centroid = chosen.mean(axis=0)
    normal = np.linalg.svd(chosen - centroid, full_matrices=False)[2][-1]
    oriented: FloatArray = normal if float(normal @ centroid) > 0 else -normal
    return oriented
