"""Mask tracks of discs: two that close on each other while their areas wobble from row to row,
sometimes a third that wanders, and arbitrary rewrites of the rows after a given one."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]

ROWS = 24
SHAPE = (40, 80)


def discs(centres: NDArray[np.float64], radii: NDArray[np.float64]) -> BoolArray:
    """(N, h, w): pixels whose centre (col + 0.5, row + 0.5) lies within each radius."""
    rows, cols = np.indices(SHAPE)
    du = cols[None] + 0.5 - centres[:, 0, None, None]
    dv = rows[None] + 0.5 - centres[:, 1, None, None]
    inside: BoolArray = du**2 + dv**2 <= radii[:, None, None] ** 2
    return inside


def approach(rng: np.random.Generator) -> BoolArray:
    """(ROWS, N, h, w): discs 0 and 1 meet near a random row between 8 and 18; masks go missing
    away from the meeting."""
    count = int(rng.integers(2, 4))
    k = np.arange(ROWS, dtype=np.float64)
    radius = rng.uniform(4.0, 7.0, count)
    meet = rng.uniform(8.0, 18.0)
    left, right = rng.uniform(8.0, 16.0), rng.uniform(64.0, 72.0)
    gap = right - left - radius[0] - radius[1]
    share = rng.uniform(0.3, 0.7)
    centres = np.empty((ROWS, count, 2))
    centres[:, 0, 0] = left + share * gap * k / meet
    centres[:, 1, 0] = right - (1.0 - share) * gap * k / meet
    centres[:, :2, 1] = 20.0 + rng.normal(0.0, 0.4, (ROWS, 2))
    if count == 3:
        start, drift = rng.uniform([10.0, 6.0], [70.0, 34.0]), rng.uniform(-1.5, 1.5, 2)
        centres[:, 2] = start + drift * k[:, None]
    radii = radius * (1.0 + rng.normal(0.0, 0.08, (ROWS, count)))
    masks = np.stack([discs(centres[t], radii[t]) for t in range(ROWS)])
    missing = (rng.random((ROWS, count)) < 0.06) & (np.abs(k - meet) > 4.0)[:, None]
    masks[missing] = False
    return masks


def rewritten_after(masks: BoolArray, row: int, rng: np.random.Generator) -> BoolArray:
    """Rows after ``row`` replaced by discs anywhere, of any size, some missing."""
    out = masks.copy()
    count = masks.shape[1]
    for t in range(row + 1, masks.shape[0]):
        centres = rng.uniform([0.0, 0.0], [SHAPE[1], SHAPE[0]], (count, 2))
        out[t] = discs(centres, np.asarray(rng.uniform(1.0, 12.0, count)))
        out[t, rng.random(count) < 0.15] = False
    return out
