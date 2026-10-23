"""The tie-aware peak index the checks share."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physverify.physics.ties import TIE_RTOL

FloatArray = NDArray[np.float64]


def first_max(values: FloatArray) -> int:
    """Index of the first value within TIE_RTOL of the largest finite one."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("first_max needs at least one finite value")
    peak = float(finite.max())
    return int(np.flatnonzero(values >= peak - TIE_RTOL * abs(peak))[0])
