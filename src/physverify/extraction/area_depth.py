"""Depth of a rigid object from the constancy of its apparent area.

A rigid object's mask area A scales as 1/z^2, so on its clean rows z_t = z_bar * sqrt(A_bar / A_t),
with z_bar and A_bar the medians of the measured depth and the area over those rows. The track keeps
its own median depth; only how depth changes from row to row comes from the area. Scaling the whole
position by z_area / z keeps the camera ray and moves the point along it; sizes and sigmas scale by
the same factor, so the angular size is preserved."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

SCALED_FIELDS = ("position", "position_sigma", "size", "size_sigma")


def mask_area(masks: BoolArray) -> FloatArray:
    """(T, N) pixel count of each (T, N, H, W) mask."""
    return np.asarray(masks, dtype=np.bool_).sum(axis=(2, 3)).astype(np.float64)


def area_depth_rows(table: ObjectStateTable, area: FloatArray) -> BoolArray:
    """(T, N) rows whose depth and area are both trusted: visible, positive finite depth, a
    non-empty mask, no nearer object over the mask (NaN cover counts as uncovered) and no border
    cut."""
    z = np.asarray(table.position[..., 2], dtype=np.float64)
    used = np.asarray(table.visible, dtype=np.bool_) & np.isfinite(z) & (z > 0) & (area > 0)
    if table.nearer_cover is not None:
        used &= ~(np.asarray(table.nearer_cover, dtype=np.float64) > 0)
    if table.border_cut is not None:
        used &= ~np.asarray(table.border_cut, dtype=np.bool_)
    return np.asarray(used, dtype=np.bool_)


def area_depth_factor(
    position_z: FloatArray, area: FloatArray, used: BoolArray, min_rows: int
) -> FloatArray:
    """(T, N) z_area / z on the used rows of tracks with at least ``min_rows`` of them; 1 on every
    other row."""
    factor = np.ones(position_z.shape, dtype=np.float64)
    for n in range(position_z.shape[1]):
        rows = used[:, n]
        if int(rows.sum()) < min_rows:
            continue
        z, a = position_z[rows, n], area[rows, n]
        factor[rows, n] = float(np.median(z)) * np.sqrt(float(np.median(a)) / a) / z
    return factor


def scale_depth_fields(table: ObjectStateTable, factor: FloatArray) -> ObjectStateTable:
    """``table`` with position, size and their sigmas multiplied by ``factor`` per row."""
    k = factor[..., None]
    return replace(table, **{name: getattr(table, name) * k for name in SCALED_FIELDS})
