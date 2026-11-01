"""Along-ray position sigma of a perceived sphere from the constancy of its apparent size.

A rigid sphere's image diameter d scales as 1/z, so over its clean rows z * d is constant up to the
depth error. ln(z * d / median(z * d)) is how far one row's depth departs from the object's own
apparent-size depth; combined with the pixel noise of d it gives a relative sigma along the ray for
that row. Positions and sizes are never changed here (ADR 0033)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.extraction.mask_border import border_cut

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SizeConstancyParams:
    min_diameter_px: float
    diameter_noise_px: float
    shape_sigma_floor: float
    min_clean_rows: int
    border_margin_px: int


def apparent_diameter(masks: BoolArray) -> FloatArray:
    """(T, N) longer side in pixels of each (T, N, H, W) mask's bounding box, NaN when empty."""
    return np.fmax(_extent(masks.any(axis=-1)), _extent(masks.any(axis=-2)))


def _extent(occupied: BoolArray) -> FloatArray:
    length = occupied.shape[-1]
    first = occupied.argmax(axis=-1)
    last = length - 1 - occupied[..., ::-1].argmax(axis=-1)
    return np.where(occupied.any(axis=-1), (last - first + 1).astype(np.float64), np.nan)


def clean_rows(
    diameter: FloatArray,
    visible: BoolArray,
    depth: FloatArray,
    cut: BoolArray,
    params: SizeConstancyParams,
) -> BoolArray:
    """Rows whose mask box measures the whole object: visible, not cut by the frame, at least
    ``min_diameter_px`` across, with a positive finite depth."""
    with np.errstate(invalid="ignore"):
        wide = diameter >= params.min_diameter_px
        ahead = depth > 0
    return np.asarray(visible & ~cut & wide & np.isfinite(depth) & ahead, dtype=np.bool_)


def constancy_log_ratio(
    depth: FloatArray, diameter: FloatArray, clean: BoolArray, min_rows: int
) -> FloatArray:
    """(T, N) ln(z * d / median(z * d)) over each object's clean rows; NaN on other rows and on
    objects with fewer than ``min_rows`` clean rows."""
    out = np.full(depth.shape, np.nan)
    for n in range(depth.shape[1]):
        rows = clean[:, n]
        if int(rows.sum()) < min_rows:
            continue
        product = depth[rows, n] * diameter[rows, n]
        out[rows, n] = np.log(product / np.median(product))
    return out


def along_ray_sigma(
    masks: BoolArray,
    depth: FloatArray,
    visible: BoolArray,
    spheres: BoolArray,
    params: SizeConstancyParams,
) -> FloatArray:
    """(T, N) relative along-ray sigma hypot(m, sqrt((e / d)^2 + f^2)) on the clean rows of the
    sphere tracks, NaN elsewhere (m the constancy log ratio, e the pixel noise of d, f the
    shape floor)."""
    diameter = apparent_diameter(masks)
    cut = border_cut(masks, params.border_margin_px)
    clean = clean_rows(diameter, visible, depth, cut, params) & spheres[None, :]
    log_ratio = constancy_log_ratio(depth, diameter, clean, params.min_clean_rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixel = np.sqrt((params.diameter_noise_px / diameter) ** 2 + params.shape_sigma_floor**2)
    return np.where(np.isfinite(log_ratio), np.hypot(log_ratio, pixel), np.nan)


def replace_along_ray_sigma(
    position: FloatArray, position_sigma: FloatArray, along: FloatArray
) -> FloatArray:
    """(T, N, 3) ``position_sigma`` with its along-ray part swapped for ``along`` * |p| per axis.

    The input's along-ray part is the relative depth sigma r = sigma_z / z spread over the axes
    as r * |p_i|; what remains of sigma_i after removing it is kept as the lateral part. Cells
    with a non-finite ``along``, depth or input sigma, or a depth <= 0, are returned unchanged."""
    sigma = np.array(position_sigma, dtype=np.float64, copy=True)
    z = position[..., 2]
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(along) & np.isfinite(z) & (z > 0) & np.isfinite(position_sigma).all(-1)
    if not usable.any():
        return sigma
    p, s, old = np.abs(position[usable]), along[usable], position_sigma[usable]
    relative = old[:, 2] / p[:, 2]
    for axis in (0, 1):
        lateral = np.maximum(old[:, axis] ** 2 - (relative * p[:, axis]) ** 2, 0.0)
        sigma[usable, axis] = np.sqrt(lateral + (s * p[:, axis]) ** 2)
    sigma[usable, 2] = s * p[:, 2]
    return sigma
