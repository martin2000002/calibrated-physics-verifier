"""One image row of an object pair over a clip: contact-zone features summarised over frames,
pooled masks and crop-encoder tokens at the reference frame.

Zone features are medians over the chosen frames, except ``zone_px_share_max``, the largest
zone share (0 when the pair never meets in the image). The reference frame is the one with the
largest zone, or the middle chosen frame when no zone appears. ``frames`` selects the frames
read: all of them for the clip row, one for the single-frame row of the shortcut control.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.extraction.contact_crops import (
    ZONE_FEATURES,
    BoolArray,
    CropSettings,
    FloatArray,
    nearest_indices,
    pair_crop,
    zone_features,
)
from physverify.extraction.crop_encoder import CropEncoder

ZONE_MAX = "zone_px_share_max"
POOL_PX = 8
MASK_PREFIXES = ("pool_a_", "pool_b_")
TOKEN_PREFIX = "token_"


@dataclass(frozen=True)
class PairFrames:
    """Per-frame arrays of a pair: (t, h, w[, 3]) images, (t, h, w) masks, depth and validity."""

    images: NDArray[np.generic]
    mask_a: BoolArray
    mask_b: BoolArray
    depth: FloatArray
    valid: BoolArray


def dense_columns() -> tuple[str, ...]:
    pooled = tuple(f"{p}{i:02d}" for p in MASK_PREFIXES for i in range(POOL_PX * POOL_PX))
    return (*ZONE_FEATURES, ZONE_MAX, *pooled)


def token_columns(width: int) -> tuple[str, ...]:
    return tuple(f"{TOKEN_PREFIX}{i:03d}" for i in range(width))


def middle_frame(n_frames: int) -> tuple[int, ...]:
    return ((n_frames - 1) // 2,)


def pooled(mask: BoolArray) -> FloatArray:
    """``POOL_PX`` x ``POOL_PX`` nearest-neighbour samples of ``mask``, flattened."""
    rows = nearest_indices(POOL_PX, mask.shape[0])
    cols = nearest_indices(POOL_PX, mask.shape[1])
    return np.asarray(mask[rows][:, cols], dtype=np.float64).reshape(-1)


def _reference(zone_share: FloatArray, frames: Sequence[int]) -> int:
    if np.nanmax(zone_share) > 0.0:
        return frames[int(np.nanargmax(zone_share))]
    return frames[(len(frames) - 1) // 2]


def pair_row(
    pair: PairFrames,
    encoder: CropEncoder,
    settings: CropSettings,
    frames: Sequence[int] | None = None,
) -> FloatArray:
    """Values in ``dense_columns()`` then ``token_columns(width)`` order."""
    chosen = list(range(len(pair.mask_a))) if frames is None else list(frames)
    per_frame = np.vstack(
        [
            zone_features(pair.mask_a[t], pair.mask_b[t], pair.depth[t], pair.valid[t], settings)
            for t in chosen
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = np.nanmedian(per_frame, axis=0) if per_frame.size else per_frame
    zone_share = per_frame[:, ZONE_FEATURES.index("zone_px_share")]
    ref = _reference(zone_share, chosen)
    crop = pair_crop(pair.images[ref], pair.mask_a[ref], pair.mask_b[ref], settings)
    tokens = encoder.encode(crop[None])[0]
    return np.concatenate(
        [
            summary,
            [float(np.nanmax(zone_share))],
            pooled(pair.mask_a[ref]),
            pooled(pair.mask_b[ref]),
            np.asarray(tokens, dtype=np.float64),
        ]
    )
