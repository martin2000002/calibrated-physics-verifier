"""Time windows given in seconds, as row counts of a state table at the clip's real frame rate:
the one conversion from seconds to rows."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from physverify.physics.tracks import row_step

# The frame rate the row-count windows were tuned at (Kubric); a window of n rows there is
# ``reference_seconds(n)``, so 24-fps tables keep their former row counts.
REFERENCE_FPS: Final = 24


def reference_seconds(rows: int) -> float:
    """Duration of ``rows`` rows at ``REFERENCE_FPS``."""
    return rows / REFERENCE_FPS


def seconds_per_row(frame_index: NDArray[np.int64], seconds_per_frame: float) -> float:
    """Real time between consecutive rows: the frame period times the native row step."""
    return seconds_per_frame * row_step(frame_index)


def rows_in(seconds: float, row_seconds: float, minimum: int = 1) -> int:
    """The row count nearest to ``seconds`` (halves round up), at least ``minimum``."""
    return max(minimum, math.floor(seconds / row_seconds + 0.5))


def odd_rows_in(seconds: float, row_seconds: float, minimum: int = 1) -> int:
    """The odd row count nearest to ``seconds``, at least the smallest odd count >= ``minimum``:
    a window centred on its row."""
    nearest = 2 * math.floor((seconds / row_seconds - 1.0) / 2.0 + 0.5) + 1
    return max(minimum + 1 - minimum % 2, nearest)
