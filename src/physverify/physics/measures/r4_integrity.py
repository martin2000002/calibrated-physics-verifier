"""R4 object integrity: missing frames [frames], size change [rel_size_change], jumps
[jump_over_size]. Inputs are the time series of a single object (time on axis 0)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from physverify.physics.finite_diff import third_difference
from physverify.physics.geometry import characteristic_length, characteristic_lengths


def missing_frames(
    visible: ArrayLike, in_frame: ArrayLike, occlusion: ArrayLike, occluded_at: float = 0.9
) -> int:
    """Frames where the object is inside the image, not visible, and not explained by an
    occlusion >= ``occluded_at``. Unknown (NaN) occlusion explains nothing."""
    shown = np.asarray(visible, dtype=bool)
    inside = np.asarray(in_frame, dtype=bool)
    occluded = np.asarray(occlusion, dtype=np.float64) >= occluded_at
    return int(np.count_nonzero(inside & ~shown & ~occluded))


def size_change_ratio(extents_t: ArrayLike) -> float:
    """max_t |L_t / median(L) - 1| over the frames with known extents (NaN if none)."""
    lengths = characteristic_lengths(np.asarray(extents_t, dtype=np.float64).reshape(-1, 3))
    known = lengths[np.isfinite(lengths)]
    if known.size == 0:
        return math.nan
    return float(np.max(np.abs(known / np.median(known) - 1.0)))


def jump_ratio(positions: ArrayLike, extents: ArrayLike, exclude: ArrayLike | None = None) -> float:
    """max_k |p(k+2) - 3 p(k+1) + 3 p(k) - p(k-1)| / (2 L), skipping rows k flagged in
    ``exclude`` (for example impacts) and stencils with unknown positions. NaN when no row
    qualifies (fewer than 4 rows).

    The third difference vanishes under any constant acceleration (flight, rest, uniform
    friction, any g), so no scale or gravity direction is needed. A step d between rows j and
    j + 1 gives |D3| = d, 2d, d at k = j - 1, j, j + 1: the measure is exactly d / L."""
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] < 4:
        return math.nan
    jumps = np.linalg.norm(third_difference(p), axis=1) / 2.0
    keep = np.isfinite(jumps)
    if exclude is not None:
        keep &= ~np.asarray(exclude, dtype=bool).reshape(-1)[1:-2]
    if not bool(np.any(keep)):
        return math.nan
    return float(np.max(jumps[keep])) / characteristic_length(extents)
