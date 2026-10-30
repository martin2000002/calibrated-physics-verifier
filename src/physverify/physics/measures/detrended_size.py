"""R4_size v3 [rel_size_change]: a size change that persists, measured against the running
median of the size ratio, so a slow drift of the ratio (depth or perspective) is not a change and
a one- or two-row flicker of a mask is not either."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from physverify.physics.running_median import running_median

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PersistentChange:
    magnitude: float
    first_row: int
    """First row of the ``persistence`` rows that hold the magnitude."""
    deviation: float
    """r / runmed(r) - 1 at the row of those that sets the magnitude (signed)."""
    row: int


def detrended_deviation(ratios: ArrayLike, window: int) -> FloatArray:
    """d_t = r_t / runmed_W(r)_t - 1 (the window shifted inside the series at its ends); NaN
    where r is unknown."""
    series = np.asarray(ratios, dtype=np.float64).reshape(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(series / running_median(series, window, "shifted") - 1.0, dtype=np.float64)


def persistent_change(ratios: ArrayLike, window: int, persistence: int) -> PersistentChange | None:
    """m = max_t min_{s in [t, t + P)} |d_s| over runs of P consecutive known rows where d keeps
    one sign (0 for a run that changes sign), with d = ``detrended_deviation``. None without any
    run of P known rows."""
    deviation = detrended_deviation(ratios, window)
    if deviation.size < persistence:
        return None
    runs = sliding_window_view(deviation, persistence)
    known = np.all(np.isfinite(runs), axis=1)
    if not bool(known.any()):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        one_sign = np.all(runs >= 0.0, axis=1) | np.all(runs <= 0.0, axis=1)
        held = np.where(one_sign, np.min(np.abs(runs), axis=1), 0.0)
    held[~known] = -math.inf
    first = int(np.argmax(held))
    row = first + int(np.argmin(np.abs(runs[first])))
    return PersistentChange(float(held[first]), first, float(deviation[row]), row)
