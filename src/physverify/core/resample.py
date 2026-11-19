"""Resizing of frames, masks, depth and intrinsics to a processing size (width, height).

Pixel convention: continuous coordinates with the image spanning [0, w] x [0, h]; resizing by
factors (sx, sy) multiplies fx and cx by sx and fy and cy by sy.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

Size = tuple[int, int]


def resize_frame(frame: NDArray[np.uint8], size: Size) -> NDArray[np.uint8]:
    """Area interpolation (anti-aliased downsampling) of one (h, w, 3) frame."""
    if (frame.shape[1], frame.shape[0]) == size:
        return frame
    return np.asarray(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)


def resize_masks(masks: NDArray[np.bool_], size: Size) -> NDArray[np.bool_]:
    """Nearest-neighbour resize of boolean masks (..., h, w)."""
    return _nearest(masks.astype(np.uint8), size).astype(bool)


def resize_depth(
    depth: NDArray[np.float32], valid: NDArray[np.bool_], size: Size
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Nearest-neighbour resize of depth (..., h, w) and its validity mask."""
    return _nearest(depth, size).astype(np.float32), resize_masks(valid, size)


def _nearest(values: NDArray[np.generic], size: Size) -> NDArray[np.generic]:
    width, height = size
    rows = _source_index(values.shape[-2], height)
    cols = _source_index(values.shape[-1], width)
    out: NDArray[np.generic] = values[..., rows[:, None], cols[None, :]]
    return out


def _source_index(native: int, target: int) -> NDArray[np.int64]:
    """Index of the native pixel whose area contains each target pixel centre."""
    centres = (np.arange(target) + 0.5) * native / target
    return np.minimum(np.floor(centres).astype(np.int64), native - 1)


def scale_intrinsics(K: NDArray[np.float64], native_size: Size, size: Size) -> NDArray[np.float64]:  # noqa: N803
    sx = size[0] / native_size[0]
    sy = size[1] / native_size[1]
    scaled: NDArray[np.float64] = np.diag([sx, sy, 1.0]) @ K
    return scaled
