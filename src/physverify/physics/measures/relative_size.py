"""R4_size v2 [rel_size_change]: an object's size relative to the other bodies of the same frame.
Two rigid bodies keep the ratio of their sizes, and an error of the depth
scale common to a frame multiplies every size of that frame alike, so it cancels in the ratio."""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def relative_lengths(lengths: ArrayLike, reference: ArrayLike) -> FloatArray:
    """Per row: L_t / exp(mean_j (log L_j,t - median_t log L_j)) over the finite positive
    references j of that row (``reference`` is (T, n_ref)); NaN where L_t or every reference is
    unknown.

    Each reference enters relative to its own median, so the denominator of a rigid scene stays 1
    when a reference leaves the image or is hidden for a while; a factor common to the row
    (depth scale) still cancels: it shifts every term alike."""
    own = np.asarray(lengths, dtype=np.float64).reshape(-1)
    others = np.asarray(reference, dtype=np.float64).reshape(own.shape[0], -1)
    usable = np.isfinite(others) & (others > 0.0)
    logs = np.where(usable, np.log(np.where(usable, others, 1.0)), math.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centred = logs - np.nanmedian(logs, axis=0)
        count = usable.sum(axis=1)
        mean = np.where(count > 0, np.nansum(centred, axis=1) / np.maximum(count, 1), math.nan)
        ratio = own / np.exp(mean)
    ratio[~(np.isfinite(own) & (own > 0.0))] = math.nan
    return np.asarray(ratio, dtype=np.float64)


def relative_size_change(lengths: ArrayLike, reference: ArrayLike) -> float:
    """max_t |r_t / median(r) - 1| for r = ``relative_lengths``; NaN without any known row."""
    ratio = relative_lengths(lengths, reference)
    known = ratio[np.isfinite(ratio)]
    if known.size == 0:
        return math.nan
    return float(np.max(np.abs(known / np.median(known) - 1.0)))
