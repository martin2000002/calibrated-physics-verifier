"""Pinhole back-projection, the one formula the lift and the floor-plane fit share."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def ray_points(u: FloatArray, v: FloatArray, z: FloatArray, intrinsics: FloatArray) -> FloatArray:
    """X = z K^-1 [u, v, 1] at continuous pixel positions; a pixel's centre is its index + 0.5."""
    x = (u - intrinsics[0, 2]) / intrinsics[0, 0] * z
    y = (v - intrinsics[1, 2]) / intrinsics[1, 1] * z
    points: FloatArray = np.column_stack([x, y, z])
    return points
