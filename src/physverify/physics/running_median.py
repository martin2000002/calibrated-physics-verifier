"""Running median of a per-row series, shared by the size check and LIFT's temporal filter."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

EdgeMode = Literal["shifted", "keep", "shrink"]


def running_median(values: ArrayLike, window: int, edges: EdgeMode) -> FloatArray:
    """Median of the finite values among ``window`` rows around each finite row (all the rows of
    a shorter series). The first and last ``window // 2`` rows use a window shifted inside the
    series (``"shifted"``) or keep their value (``"keep"``); non-finite rows are left as they
    are. ``"shrink"`` keeps every window centred on its row and narrows it near the first and
    last finite rows to the rows available on both sides, so a steady trend is not bent there."""
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if edges == "shrink":
        return _centred_median(series, window // 2)
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


def _centred_median(series: FloatArray, half: int) -> FloatArray:
    out = series.copy()
    rows = np.flatnonzero(np.isfinite(series))
    if rows.size == 0:
        return out
    first, last = int(rows[0]), int(rows[-1])
    for row in rows:
        reach = min(half, int(row) - first, last - int(row))
        span = series[row - reach : row + reach + 1]
        out[row] = float(np.median(span[np.isfinite(span)]))
    return out
