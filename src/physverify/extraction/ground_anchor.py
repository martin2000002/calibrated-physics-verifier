"""Anchoring a lift's objects to the ground plane along their viewing rays.

Convention: ``down`` is the unit gravity direction in camera coordinates (OpenCV axes). A centre p
lies down·p below the camera, so a body whose lowest point is r_v below its centre rests on a plane
h below the camera when down·p + r_v = h; along the unit ray d of p that is
lambda = h / (down·d + k) with k = r_v / |p|, a scale-free support ratio (sin of the angular
radius for a sphere).

Every row of an object implies a camera height H = down·p + r_v. A body in flight implies a smaller
H than at contact, so contact rows lie near the upper envelope (a high quantile) of H_i. A row is a
floor row when ln H is within ``z`` motion-free row-noise sigma of that envelope, ln H is steady
(its smoothed slope inside the noise band) and the caller has not excluded it (a body resting on
another one); rays flatter than ``min_elevation`` are never anchored. E_i, the median H over the
floor rows, over h, the clip's median E_i (a free unit in relative depth), is the network's depth
bias c_i. Floor rows go onto the plane (factor h / H); the object's other rows keep their depth
divided by c_i, or by c_t interpolated in time between floor rows (``per_row_smoothed``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_coeffs, savgol_filter

from physverify.physics.geometry import matrix_from_quaternion
from physverify.physics.running_median import running_median

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
MAD_TO_SIGMA = 1.4826
SCALED_FIELDS = ("position", "position_sigma", "size", "size_sigma")
SMOOTHING_ORDER = 2


class AnchorMode(StrEnum):
    """How the rows of an anchored object that are not on the floor are moved."""

    PER_ROW = "per_row"
    PER_ROW_SMOOTHED = "per_row_smoothed"


@dataclass(frozen=True)
class GroundAnchorParams:
    envelope_quantile: float
    anchor_z: float
    min_anchor_rows: int
    max_normal_sigma_rad: float
    steady_window: int
    min_elevation_rad: float
    anchor_mode: AnchorMode


class AnchorPath(StrEnum):
    """What the anchoring did to one object."""

    ANCHORED = "anchored"
    UNANCHORED = "unanchored"
    CLIP_FALLBACK = "clip_fallback"


@dataclass(frozen=True)
class AnchorResult:
    """Per-cell factor along the ray (1 = the input lift), the anchored cells, each object's bias
    c_i and path, the clip's plane height h (NaN when the whole clip falls back) and the per-cell
    ``elevation_gain``."""

    factor: FloatArray
    anchored: BoolArray
    bias: FloatArray
    paths: tuple[AnchorPath, ...]
    height: float
    gain: FloatArray


def support_half_extent(
    size: FloatArray, quaternion: FloatArray, spheres: BoolArray, down: FloatArray
) -> FloatArray:
    """r_v per (T, N) cell: half the extent along ``down`` of a box of ``size`` posed by
    ``quaternion`` (camera axes when the pose is unknown); a sphere's (N) mean radius."""
    out = np.full(size.shape[:-1], np.nan)
    for index in np.ndindex(*size.shape[:-1]):
        if spheres[index[-1]]:
            out[index] = 0.5 * float(np.mean(size[index]))
            continue
        q = quaternion[index]
        posed = bool(np.isfinite(q).all()) and float(np.linalg.norm(q)) > 0.0
        axes = matrix_from_quaternion(q) if posed else np.eye(3)
        out[index] = 0.5 * float(np.sum(size[index] * np.abs(down @ axes)))
    return out


def implied_height(position: FloatArray, support: FloatArray, down: FloatArray) -> FloatArray:
    """H = down·p + r_v: the camera height above a plane the body would touch."""
    height: FloatArray = position @ down + support
    return height


def ray_plane_distance(
    ray: FloatArray, support_ratio: FloatArray, down: FloatArray, height: float
) -> FloatArray:
    """lambda = h / (down·d + k) along unit rays d; NaN where the ray never meets the plane."""
    denominator = ray @ down + support_ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        distance: FloatArray = np.where(denominator > 0.0, height / denominator, np.nan)
    return distance


def anchor_ratio_envelope(heights: FloatArray, quantile: float) -> float:
    """Upper envelope of an object's implied heights (finite, positive rows)."""
    usable = heights[np.isfinite(heights) & (heights > 0.0)]
    return float(np.quantile(usable, quantile)) if usable.size else math.nan


def smoothed(series: FloatArray, window: int, deriv: int) -> FloatArray:
    """Savitzky-Golay (order 2) value or slope per row over the finite rows (gaps closed up);
    NaN elsewhere and when fewer than ``window`` rows are finite."""
    out = np.full(series.shape, np.nan)
    finite = np.isfinite(series)
    if int(finite.sum()) >= window:
        out[finite] = savgol_filter(
            series[finite], window, SMOOTHING_ORDER, deriv=deriv, mode="interp"
        )
    return out


def row_noise(log_heights: FloatArray, window: int) -> float:
    """Sigma of ln H without the motion: 1.4826 MAD of the residual to the smoothed series,
    divided by sqrt(1 - c0) (c0 the filter's own weight, 1/3 for 7 rows); NaN when too short."""
    residual = log_heights - smoothed(log_heights, window, 0)
    residual = residual[np.isfinite(residual)]
    if residual.size < 3:
        return math.nan
    mad = float(np.median(np.abs(residual - np.median(residual))))
    centre = float(savgol_coeffs(window, SMOOTHING_ORDER)[window // 2])
    return MAD_TO_SIGMA * mad / math.sqrt(1.0 - centre)


def steady_rows(log_heights: FloatArray, sigma: float, z: float, window: int) -> BoolArray:
    """Rows whose smoothed slope of ln H lies inside z times its own noise (sigma times the norm
    of the slope filter); rows without a slope count as steady."""
    gain = float(np.linalg.norm(savgol_coeffs(window, SMOOTHING_ORDER, deriv=1)))
    with np.errstate(invalid="ignore"):
        moving: BoolArray = np.abs(smoothed(log_heights, window, 1)) > z * sigma * gain
    return ~moving


def anchor_rows(log_heights: FloatArray, log_envelope: float, sigma: float, z: float) -> BoolArray:
    """Rows whose ln H lies within z sigma of the envelope."""
    with np.errstate(invalid="ignore"):
        near: BoolArray = np.abs(log_heights - log_envelope) <= z * sigma
    return near & np.isfinite(log_heights)


def elevation_gain(position: FloatArray, support: FloatArray, down: FloatArray) -> FloatArray:
    """|d ln lambda / d beta| = cos beta / (sin beta + k), beta the ray's elevation below the
    horizon (sin beta = down·d): the relative distance error per radian of tilt or pixel angle."""
    distance = np.linalg.norm(position, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_beta = (position @ down) / distance
        gain: FloatArray = np.sqrt(np.clip(1.0 - sin_beta**2, 0.0, None)) / (
            sin_beta + support / distance
        )
    return gain


def anchored_sigma(
    gain: FloatArray, normal_sigma: float, phi: float, pixel: FloatArray
) -> FloatArray:
    """Relative sigma of an anchored row's distance: sqrt((G sigma_g)^2 + phi^2 + (G sigma_px)^2),
    with ``pixel`` the angular sigma (rad) of the image point the ray goes through."""
    sigma: FloatArray = np.sqrt((gain * normal_sigma) ** 2 + phi**2 + (gain * pixel) ** 2)
    return sigma


def anchor_clip(
    position: FloatArray,
    size: FloatArray,
    quaternion: FloatArray,
    spheres: BoolArray,
    down: FloatArray,
    normal_sigma: float,
    pixel: FloatArray,
    excluded: BoolArray,
    params: GroundAnchorParams,
) -> AnchorResult:
    """Anchoring of a (T, N) lift; ``spheres`` flags sphere objects, ``pixel`` is the per-cell
    angular sigma of the mask centre (the row-noise floor) and ``excluded`` the cells that may
    not be floor rows."""
    n_t, n_obj = position.shape[:2]
    usable = math.isfinite(normal_sigma) and normal_sigma <= params.max_normal_sigma_rad
    if not (usable and np.isfinite(down).all()):
        return _unchanged(n_t, n_obj, AnchorPath.CLIP_FALLBACK)
    down = down / float(np.linalg.norm(down))
    support = support_half_extent(size, quaternion, spheres, down)
    heights = implied_height(position, support, down)
    gain = elevation_gain(position, support, down)
    with np.errstate(divide="ignore", invalid="ignore"):
        flat = (position @ down) / np.linalg.norm(position, axis=-1) < math.sin(
            params.min_elevation_rad
        )
    floor = gain * pixel
    contact, rows = np.full(n_obj, np.nan), np.zeros((n_t, n_obj), dtype=bool)
    for n in range(n_obj):
        rows[:, n] = _floor_rows(heights[:, n], floor[:, n], excluded[:, n] | flat[:, n], params)
        if int(rows[:, n].sum()) >= params.min_anchor_rows:
            contact[n] = float(np.median(heights[rows[:, n], n]))
    anchored_objects = np.isfinite(contact) & (contact > 0.0)
    if not anchored_objects.any():
        return _unchanged(n_t, n_obj, AnchorPath.CLIP_FALLBACK)
    height = float(np.median(contact[anchored_objects]))
    bias = np.where(anchored_objects, contact / height, 1.0)
    rows &= anchored_objects[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        on_plane = height / heights
    rest = np.ones((n_t, n_obj)) / bias[None, :]
    if params.anchor_mode is AnchorMode.PER_ROW_SMOOTHED:
        for m in np.flatnonzero(anchored_objects):
            rest[:, m] = interpolated_factor(on_plane[:, m], rows[:, m], params.steady_window)
    factor = np.where(rows, on_plane, rest)
    factor = np.where(np.isfinite(factor) & (factor > 0.0), factor, 1.0)
    paths = tuple(AnchorPath.ANCHORED if a else AnchorPath.UNANCHORED for a in anchored_objects)
    return AnchorResult(factor, rows, bias, paths, height, gain)


def _floor_rows(
    heights: FloatArray, floor: FloatArray, excluded: BoolArray, params: GroundAnchorParams
) -> BoolArray:
    envelope = anchor_ratio_envelope(heights, params.envelope_quantile)
    none = np.zeros(heights.shape, dtype=bool)
    if not math.isfinite(envelope) or envelope <= 0.0:
        return none
    with np.errstate(divide="ignore", invalid="ignore"):
        log_heights = np.where(heights > 0.0, np.log(heights), np.nan)
    floors = floor[np.isfinite(log_heights) & np.isfinite(floor)]
    noise = np.nanmax(
        [
            row_noise(log_heights, params.steady_window),
            float(np.median(floors)) if floors.size else 0.0,
        ]
    )
    if not math.isfinite(noise):
        return none
    near = anchor_rows(log_heights, math.log(envelope), noise, params.anchor_z)
    steady = steady_rows(log_heights, noise, params.anchor_z, params.steady_window)
    return near & steady & ~excluded


def interpolated_factor(on_plane: FloatArray, rows: BoolArray, window: int) -> FloatArray:
    """1 / c_t: ln c_t = -ln(h / H) on the floor rows, a running median over them, linear in time
    between them and held beyond the first and last."""
    index = np.flatnonzero(rows)
    log_bias = running_median(-np.log(on_plane[index]), window, "shifted")
    times = np.arange(on_plane.size, dtype=np.float64)
    return np.asarray(np.exp(-np.interp(times, index.astype(np.float64), log_bias)))


def _unchanged(n_t: int, n_obj: int, path: AnchorPath) -> AnchorResult:
    return AnchorResult(
        factor=np.ones((n_t, n_obj)),
        anchored=np.zeros((n_t, n_obj), dtype=bool),
        bias=np.ones(n_obj),
        paths=(path,) * n_obj,
        height=math.nan,
        gain=np.full((n_t, n_obj), np.nan),
    )


def anchored_fields(
    fields: dict[str, FloatArray],
    result: AnchorResult,
    sigma_rel: FloatArray,
    depth_rel: float,
) -> dict[str, FloatArray]:
    """Position, size and their sigma scaled by each cell's factor (the lift's placement and
    closed-form sigma are linear in the distance); on anchored cells the depth term p_k r_d of the
    closed form becomes the anchor's p_k sigma_rel (variance swap)."""
    factor = result.factor[..., None]
    out = {name: np.asarray(fields[name], dtype=np.float64) * factor for name in SCALED_FIELDS}
    swap = sigma_rel[result.anchored, None] ** 2 - depth_rel**2
    for value, sigma in (("position", "position_sigma"), ("size", "size_sigma")):
        cells = out[sigma][result.anchored]
        variance = cells**2 + out[value][result.anchored] ** 2 * swap
        out[sigma][result.anchored] = np.sqrt(np.clip(variance, 0.0, None))
    return out
