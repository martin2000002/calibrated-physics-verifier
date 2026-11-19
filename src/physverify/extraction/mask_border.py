"""Rows where a tracked mask reaches the image border: the object is cut by the frame there, so
its mask extents describe the visible part, not the object."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]


def border_cut(masks: BoolArray, margin_px: int) -> BoolArray:
    """(T, N) True where a (T, N, H, W) mask has a pixel in the outer ``margin_px`` rows or
    columns of the image."""
    m = margin_px
    return np.asarray(
        masks[..., :m, :].any(axis=(-2, -1))
        | masks[..., -m:, :].any(axis=(-2, -1))
        | masks[..., :, :m].any(axis=(-2, -1))
        | masks[..., :, -m:].any(axis=(-2, -1)),
        dtype=np.bool_,
    )
