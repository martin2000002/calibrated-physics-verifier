"""Finite differences on possibly non-uniform time grids; closed-form sigma of the third (0002).

Arrays are indexed by time on axis 0. Both difference operators are exact for quadratics in
time and return NaN on the first and last sample, where the centred stencil does not exist.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _stencil(x: ArrayLike, t: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    values = np.asarray(x, dtype=np.float64)
    times = np.asarray(t, dtype=np.float64)
    if times.ndim != 1 or values.shape[:1] != times.shape:
        raise ValueError("x must have the length of t on axis 0")
    if times.shape[0] and bool(np.any(np.diff(times) <= 0)):
        raise ValueError("t must be strictly increasing")
    extra = (1,) * (values.ndim - 1)
    h_minus = (times[1:-1] - times[:-2]).reshape(-1, *extra)
    h_plus = (times[2:] - times[1:-1]).reshape(-1, *extra)
    return values, h_minus, h_plus


def _pad(values: FloatArray, interior: FloatArray) -> FloatArray:
    out = np.full(values.shape, np.nan)
    if values.shape[0] >= 3:
        out[1:-1] = interior
    return out


def central_difference(x: ArrayLike, t: ArrayLike) -> FloatArray:
    """First derivative: weighted three-point stencil (exact for quadratics)."""
    values, hm, hp = _stencil(x, t)
    if values.shape[0] < 3:
        return np.full(values.shape, np.nan)
    prev, here, nxt = values[:-2], values[1:-1], values[2:]
    interior = (hm**2 * nxt - hp**2 * prev + (hp**2 - hm**2) * here) / (hm * hp * (hm + hp))
    return _pad(values, interior)


def second_difference(x: ArrayLike, t: ArrayLike) -> FloatArray:
    """Second derivative: 2 [h- x+ - (h- + h+) x + h+ x-] / (h- h+ (h- + h+))."""
    values, hm, hp = _stencil(x, t)
    if values.shape[0] < 3:
        return np.full(values.shape, np.nan)
    prev, here, nxt = values[:-2], values[1:-1], values[2:]
    interior = 2.0 * (hm * nxt - (hm + hp) * here + hp * prev) / (hm * hp * (hm + hp))
    return _pad(values, interior)


def third_difference(x: ArrayLike) -> FloatArray:
    """Unscaled third difference on uniformly spaced rows, one value per stencil k-1..k+2:
    x(k+2) - 3 x(k+1) + 3 x(k) - x(k-1). Zero for any quadratic; empty below 4 rows."""
    values = np.asarray(x, dtype=np.float64)
    if values.shape[0] < 4:
        return np.empty((0, *values.shape[1:]), dtype=np.float64)
    return np.asarray(values[3:] - 3.0 * values[2:-1] + 3.0 * values[1:-2] - values[:-3])


def third_difference_sigma(sigma_x: float, dt: float) -> float:
    """Sigma of (x(k+2) - 3x(k+1) + 3x(k) - x(k-1)) / dt^3 for independent position noise:
    sqrt(1 + 9 + 9 + 1) sigma_x / dt^3 = sqrt(20) sigma_x / dt^3."""
    return math.sqrt(20.0) * sigma_x / dt**3
