"""Running median of a per-row series, shared by the size check and LIFT's temporal filter."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

EdgeMode = Literal["shifted", "keep"]


def running_median(values: ArrayLike, window: int, edges: EdgeMode) -> FloatArray:
    """Median of the finite values among ``window`` rows around each finite row (all the rows of
    a shorter series). The first and last ``window // 2`` rows use a window shifted inside the
    series (``"shifted"``) or keep their value (``"keep"``); non-finite rows are left as they
    are."""
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    out = series.copy()
    n = series.size
    width = min(window, n)
    half = window // 2
    for row in np.flatnonzero(np.isfinite(series)):
        if edges == "keep" and not half <= row < n - half:
            continue
        first = min(max(int(row) - half, 0), n - width)
        span = series[first : first + width]
        out[row] = float(np.median(span[np.isfinite(span)]))
    return out
