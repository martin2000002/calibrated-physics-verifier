"""Rows where a body rests on another one, seen in the image: its lowest mask pixels along the
projected gravity, moved a few pixels further down, land in another object's mask."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
BOTTOM_BAND_PX = 1.5
"""Mask pixels this close to the lowest one (along the image gravity) form the bottom edge."""
PROBE_STEP = 0.05
"""Length of the 3D gravity step projected to find the image gravity, in units of the distance."""


def image_down(
    intrinsics: FloatArray, position: FloatArray, down: FloatArray, scale: FloatArray
) -> FloatArray | None:
    """Unit image direction (x, y in mask pixels) of gravity at a body's centre; None when the
    centre or the probe is not in front of the camera."""
    if not np.isfinite(position).all() or position[2] <= 0.0:
        return None
    probe = position + PROBE_STEP * float(np.linalg.norm(position)) * down
    if probe[2] <= 0.0:
        return None
    start, end = intrinsics @ position, intrinsics @ probe
    step = (end[:2] / end[2] - start[:2] / start[2]) * scale
    norm = float(np.linalg.norm(step))
    return step / norm if norm > 0.0 else None


def bottom_pixels(mask: BoolArray, direction: FloatArray) -> FloatArray | None:
    """(x, y) of the mask pixels within BOTTOM_BAND_PX of its extreme along ``direction``."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    reach = xs * direction[0] + ys * direction[1]
    edge = reach >= float(reach.max()) - BOTTOM_BAND_PX
    return np.column_stack([xs[edge], ys[edge]]).astype(np.float64)


def touches_below(pixels: FloatArray, direction: FloatArray, others: BoolArray, reach: int) -> bool:
    """Whether ``pixels`` moved 1..reach px along ``direction`` meet a True cell of ``others``."""
    height, width = others.shape
    for step in range(1, reach + 1):
        probe = np.rint(pixels + step * direction[None, :]).astype(np.int64)
        inside = (probe[:, 0] >= 0) & (probe[:, 0] < width) & (probe[:, 1] >= 0)
        inside &= probe[:, 1] < height
        if bool(others[probe[inside, 1], probe[inside, 0]].any()):
            return True
    return False


def stacked_rows(
    masks: BoolArray,
    position: FloatArray,
    intrinsics: FloatArray,
    down: FloatArray,
    scale: FloatArray,
    reach: int,
) -> BoolArray:
    """(T, N) cells whose body sits on another tracked body; ``scale`` maps the intrinsics'
    pixels to mask pixels (x, y)."""
    n_t, n_obj = masks.shape[:2]
    out = np.zeros((n_t, n_obj), dtype=bool)
    if not np.isfinite(down).all():
        return out
    for t in range(n_t):
        union = masks[t].any(axis=0)
        for n in range(n_obj):
            direction = image_down(intrinsics, position[t, n], down, scale)
            pixels = None if direction is None else bottom_pixels(masks[t, n], direction)
            if direction is not None and pixels is not None:
                out[t, n] = touches_below(pixels, direction, union & ~masks[t, n], reach)
    return out
