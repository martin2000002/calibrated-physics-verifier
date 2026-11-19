"""Extents at the object's windowed depth. A lift's extent is e_px(t) z(t) / f, so the per-row
noise of a monocular depth z enters every extent; rescaling by runmed_W(z) / z keeps the
silhouette e_px(t) row by row and takes z from the running median of the object's own depth."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.physics.running_median import running_median

FloatArray = NDArray[np.float64]


def window_depth_factor(depth: ArrayLike, visible: ArrayLike, window: int) -> FloatArray:
    """(T, N) runmed_W(z)_t / z_t per object, over the visible rows with a finite depth (the
    window shifted inside the series at its ends); 1 where that ratio is unknown or not
    positive, so those rows keep their extents."""
    z = np.asarray(depth, dtype=np.float64)
    z = np.where(np.asarray(visible, dtype=np.bool_) & np.isfinite(z), z, np.nan)
    smooth = np.empty_like(z)
    for index in range(z.shape[1]):
        smooth[:, index] = running_median(z[:, index], window, "shifted")
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = smooth / z
    return np.where(np.isfinite(factor) & (factor > 0.0), factor, 1.0)
