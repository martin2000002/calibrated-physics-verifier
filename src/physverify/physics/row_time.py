"""Time windows given in seconds, as row counts of a state table at the clip's real frame rate:
the one conversion from seconds to rows."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from physverify.physics.tracks import row_step


def seconds_per_row(frame_index: NDArray[np.int64], seconds_per_frame: float) -> float:
    """Real time between consecutive rows: the frame period times the native row step."""
    return seconds_per_frame * row_step(frame_index)


def rows_in(seconds: float, row_seconds: float) -> int:
    """The row count nearest to ``seconds`` (halves round up), at least 1."""
    return max(1, math.floor(seconds / row_seconds + 0.5))


def odd_rows_in(seconds: float, row_seconds: float) -> int:
    """The odd row count nearest to ``seconds``, at least 1: a window centred on its row."""
    return max(1, 2 * math.floor((seconds / row_seconds - 1.0) / 2.0 + 0.5) + 1)
