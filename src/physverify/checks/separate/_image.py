"""Pinhole projection of camera-frame states (OpenCV axes: x right, y down, z forward)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physverify.checks._base import CheckInputs
from physverify.physics.contact_manifold import convex_hull_indices
from physverify.schema.states import CoordFrame

FloatArray = NDArray[np.float64]


def camera_k(inputs: CheckInputs) -> FloatArray | None:
    """K of the camera the states were lifted with, None without one or off the camera frame."""
    if inputs.camera is None or inputs.states.frame is not CoordFrame.CAMERA_OPENCV:
        return None
    return np.asarray(inputs.camera.K, dtype=np.float64)


def project(k: FloatArray, points: FloatArray) -> FloatArray:
    """Pixels (..., 2) of camera-frame points (..., 3)."""
    homogeneous = points @ k.T
    return np.asarray(homogeneous[..., :2] / homogeneous[..., 2:3], dtype=np.float64)


def row_sigma(k: FloatArray, point: FloatArray, sigma: FloatArray) -> float:
    """Delta-method sigma of the image row of ``point`` with independent per-axis ``sigma``:
    J_v = (K[1] - v K[2]) / z."""
    v = float(project(k, point)[1])
    jacobian = (k[1] - v * k[2]) / float(point[2])
    return float(np.sqrt(np.sum((jacobian * sigma) ** 2)))


def outline(k: FloatArray, points: FloatArray) -> FloatArray:
    """Counter-clockwise convex hull (n, 2) of the projected camera-frame ``points``."""
    pixels = project(k, points)
    return pixels[convex_hull_indices(pixels)]


def overlapping(a: FloatArray, b: FloatArray) -> bool:
    """Whether two convex polygons share interior points (separating-axis test)."""
    for polygon in (a, b):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for normal in np.stack([edges[:, 1], -edges[:, 0]], axis=1):
            pa, pb = a @ normal, b @ normal
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    return True
