"""Depth to camera-frame geometry: shared by LIFT, depth ground truth and the extractor diagnosis.

Pixel centres are (col + 0.5, row + 0.5), the convention of the synthetic fixture and of K.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_erosion

from physverify.physics.geometry import characteristic_length
from physverify.physics.uncertainty import entry_draws, entry_rms, monte_carlo
from physverify.schema.error_profile import ErrorEntry
from physverify.schema.perception import DepthKind, DepthMap, MaskTrack
from physverify.schema.states import LengthUnit

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DISPARITY_EPS = 1e-6
MIN_REQ_RANGE_REL = 0.1
"""size_constancy needs some object whose apparent radius changes by at least 10 %."""


class ShiftStrategy(StrEnum):
    ZERO_SHIFT = "zero_shift"
    SIZE_CONSTANCY = "size_constancy"


def normalize_by_object_median(
    z: FloatArray, valid: BoolArray, objects: BoolArray
) -> tuple[FloatArray, float]:
    """z divided by its median over valid object pixels of the whole clip (else all valid)."""
    pool = z[valid & objects]
    if pool.size == 0:
        pool = z[valid]
    if pool.size == 0:
        raise ValueError("no valid depth pixel to normalize by")
    median = float(np.median(pool))
    if not (math.isfinite(median) and median > 0):
        raise ValueError(f"object depth median must be finite and > 0, got {median}")
    return z / median, median


def _object_disparity_samples(
    depth: DepthMap, masks: MaskTrack
) -> list[tuple[FloatArray, FloatArray]]:
    """Per object: (median disparity, apparent radius r_eq = sqrt(area / pi)) per usable frame."""
    samples = []
    for n in range(masks.masks.shape[1]):
        d_hat, r_eq = [], []
        for t in range(masks.masks.shape[0]):
            pixels = masks.masks[t, n] & depth.valid[t]
            if pixels.sum() == 0:
                continue
            d_hat.append(float(np.median(depth.depth[t][pixels])))
            r_eq.append(math.sqrt(float(masks.masks[t, n].sum()) / math.pi))
        if len(d_hat) >= 2:
            samples.append((np.array(d_hat), np.array(r_eq)))
    return samples


def resolve_disparity_shift(depth: DepthMap, masks: MaskTrack, strategy: str) -> float:
    """Shift b of an affine disparity d = a / z + b.

    ``size_constancy``: a rigid object keeps r_eq z constant, so d_i(t) = b + c_i r_eq,i(t);
    b is shared, c_i per object, fitted by least squares. Falls back to 0 when no object changes
    apparent size by >= 10 % or when b would reach an object's disparity."""
    if ShiftStrategy(strategy) is ShiftStrategy.ZERO_SHIFT:
        return 0.0
    samples = _object_disparity_samples(depth, masks)
    moving = [np.ptp(r) / np.median(r) >= MIN_REQ_RANGE_REL for _, r in samples]
    if not any(moving):
        return 0.0
    rows = sum(len(d) for d, _ in samples)
    design = np.zeros((rows, 1 + len(samples)))
    design[:, 0] = 1.0
    target = np.concatenate([d for d, _ in samples])
    start = 0
    for i, (_, r_eq) in enumerate(samples):
        design[start : start + r_eq.size, 1 + i] = r_eq
        start += r_eq.size
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    shift = float(solution[0])
    if not math.isfinite(shift) or shift >= float(target.min()):
        return 0.0
    return shift


def lift_depth(
    depth: DepthMap, masks: MaskTrack, strategy: str
) -> tuple[FloatArray, BoolArray, LengthUnit]:
    """Depth along the optical axis: metres for metric z, else clip-relative (object median 1)."""
    valid = depth.valid & np.isfinite(depth.depth)
    z = depth.depth.astype(np.float64)
    if depth.kind is DepthKind.METRIC_Z:
        valid &= z > 0
        return z, valid, LengthUnit.METER
    objects = masks.masks.any(axis=1)
    if depth.kind is DepthKind.AFFINE_DISPARITY:
        shifted = z - resolve_disparity_shift(depth, masks, strategy)
        valid &= shifted > DISPARITY_EPS
        z = 1.0 / np.maximum(shifted, DISPARITY_EPS)
    else:
        valid &= z > 0
    relative, _ = normalize_by_object_median(z, valid, objects)
    return relative, valid, LengthUnit.RELATIVE


@dataclass(frozen=True)
class ObjectSummary:
    """Median camera-frame point ``centre`` of the visible surface and full ``extents``."""

    centre: FloatArray
    extents: FloatArray
    r_eq_px: float
    n_pixels: int


@dataclass(frozen=True)
class SummaryOptions:
    erode_px: int
    extent_percentiles: tuple[float, float]
    min_area_px: int


def _surface_pixels(mask: BoolArray, valid: BoolArray, erode_px: int) -> BoolArray:
    eroded = binary_erosion(mask, iterations=erode_px) if erode_px > 0 else mask
    pixels = eroded & valid
    return pixels if bool(pixels.any()) else mask & valid


def object_summary(
    mask: BoolArray,
    z: FloatArray,
    valid: BoolArray,
    intrinsics: FloatArray,
    options: SummaryOptions,
) -> ObjectSummary | None:
    """Back-projects X = z K^-1 [u, v, 1] of the eroded surface for the centre s; None when fewer
    than ``min_area_px`` pixels remain.

    Extents come from the whole silhouette: (P_hi - P_lo of the pixel coordinates + 1 px) z_s / f.
    The eroded set is only for a depth median free of edge bleeding; on objects of a few dozen
    pixels it would shrink the extents by a third."""
    pixels = _surface_pixels(mask, valid, options.erode_px)
    n_pixels = int(pixels.sum())
    if n_pixels < options.min_area_px:
        return None
    rows, cols = np.nonzero(pixels)
    depth = z[rows, cols]
    x = (cols + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0] * depth
    y = (rows + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1] * depth
    centre = np.median(np.column_stack([x, y, depth]), axis=0)
    e_xy = silhouette_spread_px(mask, options.extent_percentiles) * centre[2]
    e_xy = e_xy / np.array([intrinsics[0, 0], intrinsics[1, 1]])
    return ObjectSummary(
        centre=centre,
        extents=full_extents(e_xy),
        r_eq_px=math.sqrt(float(mask.sum()) / math.pi),
        n_pixels=n_pixels,
    )


def full_extents(lateral: FloatArray) -> FloatArray:
    """(e_x, e_y, e_z) from the lateral extents (..., 2): e_z = (e_x + e_y) / 2 is not observable
    from one view, so it is assumed, not measured."""
    lateral = np.asarray(lateral, dtype=np.float64)
    extents: FloatArray = np.concatenate([lateral, lateral.mean(axis=-1, keepdims=True)], axis=-1)
    return extents


def silhouette_spread_px(mask: BoolArray, percentiles: tuple[float, float]) -> FloatArray:
    """(width, height) in pixels: percentile spread of pixel centres plus one pixel."""
    rows, cols = np.nonzero(mask)
    uv = np.column_stack([cols, rows]).astype(np.float64)
    lo, hi = percentiles
    spread: FloatArray = np.percentile(uv, hi, axis=0) - np.percentile(uv, lo, axis=0) + 1.0
    return spread


def object_position(centre: FloatArray, extents: FloatArray, offset_frac: float) -> FloatArray:
    """p = s + k e_z s_hat: the median surface point pushed back along the viewing ray."""
    ray = centre / np.linalg.norm(centre, axis=-1, keepdims=True)
    position: FloatArray = centre + offset_frac * extents[..., 2:3] * ray
    return position


@dataclass(frozen=True)
class LiftErrors:
    """Profile entries read by LIFT (None = unknown) and the relative focal error phi."""

    depth_rel: ErrorEntry | None
    centroid_px_rel: ErrorEntry | None
    centroid_rel: ErrorEntry | None
    size_rel: ErrorEntry | None
    phi: float


def focal_rel_sigma(fov_deg: float, fov_sigma_deg: float) -> float:
    """phi = sigma_fov / sin(fov), from f = (W/2) / tan(fov/2)."""
    return math.radians(fov_sigma_deg) / math.sin(math.radians(fov_deg))


def closed_form_sigma(
    position: FloatArray,
    extents: FloatArray,
    r_eq_px: float,
    intrinsics: FloatArray,
    errors: LiftErrors,
) -> tuple[FloatArray, FloatArray]:
    """Position and extent sigma per camera axis; NaN when an entry is unknown."""
    r_d = entry_rms(errors.depth_rel)
    c = entry_rms(errors.centroid_px_rel)
    ell = entry_rms(errors.centroid_rel)
    q = entry_rms(errors.size_rel)
    phi = errors.phi
    size = characteristic_length(extents)
    x, y, z = position
    focal = np.array([intrinsics[0, 0], intrinsics[1, 1]])
    lateral = np.array([x, y])
    sigma_xy = np.sqrt(
        (lateral * r_d) ** 2
        + (z * c * r_eq_px / focal) ** 2
        + (lateral * phi) ** 2
        + (ell * size) ** 2
    )
    sigma_z = math.sqrt((z * r_d) ** 2 + (ell * size) ** 2)
    # e_z = (e_x + e_y) / 2 inherits the focal error of both lateral extents.
    size_sigma = extents * math.sqrt(r_d**2 + phi**2 + q**2)
    return np.array([*sigma_xy, sigma_z]), size_sigma


def focal_draws(phi: float, rng: np.random.Generator, n_draws: int) -> FloatArray | None:
    """Relative focal errors; None (unknown, so every sampled sigma is NaN) when phi is NaN."""
    if math.isnan(phi):
        return None
    return rng.normal(0.0, phi, size=(n_draws,)) if phi > 0 else np.zeros(n_draws)


def monte_carlo_sigma(
    summary: ObjectSummary,
    intrinsics: FloatArray,
    offset_frac: float,
    errors: LiftErrors,
    rng: np.random.Generator,
    n_draws: int,
) -> tuple[FloatArray, FloatArray]:
    """Std of position and extents over draws of (depth, image centroid, focal, own) errors."""
    focal = np.array([intrinsics[0, 0], intrinsics[1, 1]])
    principal = np.array([intrinsics[0, 2], intrinsics[1, 2]])
    z_s = float(summary.centre[2])
    pixel = principal + focal * summary.centre[:2] / z_s
    draws: Mapping[str, FloatArray | None] = {
        "depth": entry_draws(errors.depth_rel, rng, (n_draws,)),
        "centroid_px": entry_draws(errors.centroid_px_rel, rng, (n_draws, 2)),
        "centroid": entry_draws(errors.centroid_rel, rng, (n_draws, 3)),
        "size": entry_draws(errors.size_rel, rng, (n_draws, 3)),
        "focal": focal_draws(errors.phi, rng, n_draws),
    }
    size = characteristic_length(summary.extents)

    def columns(values: FloatArray, k: int) -> FloatArray:
        return np.broadcast_to(np.reshape(values, (n_draws, -1)), (n_draws, k))

    def model(d: Mapping[str, FloatArray]) -> FloatArray:
        depth_ratio = 1.0 + d["depth"]
        f_ratio = 1.0 + d["focal"]
        uv = pixel + columns(d["centroid_px"], 2) * summary.r_eq_px
        xy = (z_s * depth_ratio)[:, None] * (uv - principal) / (focal * f_ratio[:, None])
        centre = np.column_stack([xy, z_s * depth_ratio])
        extents = summary.extents * (depth_ratio / f_ratio)[:, None] * (1.0 + columns(d["size"], 3))
        position = object_position(centre, extents, offset_frac)
        position = position + columns(d["centroid"], 3) * size
        return np.column_stack([position, extents])

    if all(value is None for value in draws.values()):
        return np.full(3, np.nan), np.full(3, np.nan)
    result = monte_carlo(model, draws)
    return result.std[:3], result.std[3:]
