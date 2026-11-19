"""Rows whose third-difference stencil spans a step of the object's size: a rigid body keeps its
size, so such a step is the mask or the depth changing, not the body moving."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.physics.geometry import characteristic_lengths

BoolArray = NDArray[np.bool_]


def size_steps(extents: ArrayLike) -> NDArray[np.float64]:
    """(T - 1,) |L_{j+1} / L_j - 1| between consecutive rows, L = (e_x e_y e_z)^(1/3); NaN where
    either length is unknown or not positive."""
    lengths = characteristic_lengths(extents)
    with np.errstate(invalid="ignore", divide="ignore"):
        steps = np.abs(lengths[1:] / lengths[:-1] - 1.0)
    known = (lengths[1:] > 0.0) & (lengths[:-1] > 0.0)
    return np.where(known, steps, math.nan)


def size_step_stencil_mask(extents: ArrayLike, max_step: float) -> BoolArray:
    """Rows k whose stencil (rows k-1..k+2, as ``impact_stencil_mask``) holds a step above
    ``max_step``; an unknown step never flags a row."""
    steps = size_steps(extents)
    big = np.nan_to_num(steps, nan=0.0) > max_step
    padded = np.concatenate([[False], big, [False, False]])
    return np.asarray(padded[:-2] | padded[1:-1] | padded[2:], dtype=np.bool_)
