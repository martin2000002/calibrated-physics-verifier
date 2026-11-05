"""Image-side features of an object pair at their contact zone, from masks, depth and frames.

The contact zone of a frame is where the two masks, each dilated by ``reach_px``, overlap. Its
centroid anchors a square crop of the frame; when the dilated masks do not meet, the crop sits
halfway between the mask centroids and the zone features are zero. Depth is read on the zone
(core), on a ring of ``ring_px`` around it and on each mask near the zone, as medians of the
valid pixels, so a reading is NaN when no valid depth falls there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]

ZONE_FEATURES = (
    "zone_px_share",
    "overlap_px_share",
    "visible_a_share",
    "visible_b_share",
    "depth_core",
    "depth_ring",
    "depth_gap_ab",
)


@dataclass(frozen=True)
class CropSettings:
    size_px: int = 64
    reach_px: int = 3
    ring_px: int = 4


def _dilate(mask: BoolArray, px: int) -> BoolArray:
    from scipy.ndimage import binary_dilation

    if px <= 0:
        return mask.copy()
    return np.asarray(binary_dilation(mask, iterations=px), dtype=bool)


def contact_zone(mask_a: BoolArray, mask_b: BoolArray, reach_px: int) -> BoolArray:
    return _dilate(mask_a, reach_px) & _dilate(mask_b, reach_px)


def _centroid(mask: BoolArray) -> tuple[float, float] | None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    return float(rows.mean()), float(cols.mean())


def crop_centre(mask_a: BoolArray, mask_b: BoolArray, zone: BoolArray) -> tuple[float, float]:
    centre = _centroid(zone)
    if centre is not None:
        return centre
    ca, cb = _centroid(mask_a), _centroid(mask_b)
    known = [c for c in (ca, cb) if c is not None]
    if not known:
        return (mask_a.shape[0] - 1) / 2, (mask_a.shape[1] - 1) / 2
    return float(np.mean([c[0] for c in known])), float(np.mean([c[1] for c in known]))


def square_crop(image: NDArray[np.generic], centre: tuple[float, float], size: int) -> FloatArray:
    """``size`` x ``size`` window of ``image`` (h, w[, c]) centred on ``centre``, edge-padded."""
    half = size // 2
    pad = [(half, size - half)] * 2 + [(0, 0)] * (image.ndim - 2)
    padded = np.pad(np.asarray(image, dtype=np.float64), pad, mode="edge")
    row, col = round(centre[0]), round(centre[1])
    row = min(max(row, 0), image.shape[0] - 1)
    col = min(max(col, 0), image.shape[1] - 1)
    return padded[row : row + size, col : col + size]


def _median(values: FloatArray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def zone_features(
    mask_a: BoolArray,
    mask_b: BoolArray,
    depth: FloatArray,
    valid: BoolArray,
    settings: CropSettings,
) -> FloatArray:
    """Values in ``ZONE_FEATURES`` order for one frame; shares are over the image area."""
    area = float(mask_a.size)
    zone = contact_zone(mask_a, mask_b, settings.reach_px)
    ring = _dilate(zone, settings.ring_px) & ~zone
    usable = np.where(valid, depth.astype(np.float64), np.nan)
    near = _dilate(zone, settings.reach_px)
    near_a = usable[near & mask_a] if (near & mask_a).any() else usable[mask_a]
    near_b = usable[near & mask_b] if (near & mask_b).any() else usable[mask_b]
    return np.array(
        [
            zone.sum() / area,
            (mask_a & mask_b).sum() / area,
            mask_a.sum() / area,
            mask_b.sum() / area,
            _median(usable[zone]),
            _median(usable[ring]),
            _median(near_a) - _median(near_b),
        ]
    )


def pair_crop(
    frame: NDArray[np.generic], mask_a: BoolArray, mask_b: BoolArray, settings: CropSettings
) -> FloatArray:
    """Crop of ``frame`` at the pair's contact zone, resampled to the mask resolution first."""
    image = _resize_to(frame, mask_a.shape)
    zone = contact_zone(mask_a, mask_b, settings.reach_px)
    return square_crop(image, crop_centre(mask_a, mask_b, zone), settings.size_px)


def nearest_indices(n_out: int, n_in: int) -> NDArray[np.int64]:
    """Source index of each of ``n_out`` samples of an axis of ``n_in`` (nearest neighbour)."""
    return np.minimum((np.arange(n_out) * n_in / n_out).astype(np.int64), n_in - 1)


def _resize_to(frame: NDArray[np.generic], shape: tuple[int, ...]) -> NDArray[np.generic]:
    if frame.shape[:2] == shape:
        return frame
    rows = nearest_indices(shape[0], frame.shape[0])
    cols = nearest_indices(shape[1], frame.shape[1])
    resized: NDArray[np.generic] = frame[rows][:, cols]
    return resized
