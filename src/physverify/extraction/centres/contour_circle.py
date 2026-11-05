"""A sphere's centre from the visible part of its mask contour, with no depth.

A sphere's silhouette is close to a disc. Its centre comes from a circle of known radius fitted to
the contour pixels that are silhouette: contour pixels within ``exclusion_px`` of another tracked
mask (where bodies touch or one hides the other) or of the image border are left out. The radius
is the track's own, sqrt(A / pi) on nearby rows whose mask touches nothing. The centres of a
digitised disc's contour pixels lie about half a pixel inside its edge, so the fitted circle has
radius R - ``boundary_inset_px``. With too little silhouette the mask centroid is used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from pydantic import Field
from scipy.ndimage import binary_dilation, binary_erosion

from physverify.extraction.centres._base import (
    BoolArray,
    CentreMethod,
    CentreParams,
    FloatArray,
    ImageCentres,
)
from physverify.schema.states import PrimitiveKind

# Solver settings, not physics: Gauss-Newton converges in a few steps on a contour arc.
FIT_ITERATIONS = 20
FIT_STEP_TOL_PX = 1e-6
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
SQUARE = np.ones((3, 3), dtype=bool)


class Params(CentreParams):
    """Fixed a priori, never fitted on calibration data."""

    area_min_px: int = Field(default=3, ge=1)
    contour_min_px: int = Field(default=5, ge=2)
    boundary_inset_px: float = Field(default=0.5, ge=0.0)
    exclusion_px: int = Field(default=1, ge=1)
    radius_window_rows: int = Field(default=6, ge=0)
    min_arc_deg: float = Field(default=90.0, gt=0.0, le=360.0)
    primitives: tuple[PrimitiveKind, ...] = (PrimitiveKind.SPHERE,)


def pixel_uv(mask: BoolArray) -> FloatArray:
    rows, cols = np.nonzero(mask)
    return np.column_stack([cols + 0.5, rows + 0.5]).astype(np.float64)


def blocked(others: BoolArray, exclusion_px: int) -> BoolArray:
    """Pixels within ``exclusion_px`` (8-neighbourhood) of another mask or of the image border."""
    near: BoolArray = np.zeros(others.shape, dtype=bool)
    if others.any():
        near = binary_dilation(others, structure=SQUARE, iterations=exclusion_px)
    k = exclusion_px
    near[:k, :] = near[-k:, :] = near[:, :k] = near[:, -k:] = True
    return near


def silhouette_contour(mask: BoolArray, block: BoolArray) -> FloatArray:
    """Contour pixels (4-connected boundary) outside ``block``."""
    edge = mask & ~binary_erosion(mask, structure=CROSS, border_value=0)
    return pixel_uv(edge & ~block)


def angular_span_deg(points: FloatArray, centre: FloatArray) -> float:
    """360 degrees minus the largest angular gap of the points seen from ``centre``."""
    if len(points) < 2:
        return 0.0
    a = np.sort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))
    gaps = np.diff(np.concatenate([a, a[:1] + 2.0 * math.pi]))
    return math.degrees(2.0 * math.pi - float(gaps.max()))


def fit_known_radius(points: FloatArray, rho: float, start: FloatArray) -> FloatArray | None:
    """Gauss-Newton for min sum (|p - c| - rho)^2 over the centre c."""
    c = np.asarray(start, dtype=np.float64).copy()
    for _ in range(FIT_ITERATIONS):
        diff = points - c
        dist = np.linalg.norm(diff, axis=1)
        if np.any(dist == 0.0):
            return None
        jac = -diff / dist[:, None]
        step, *_ = np.linalg.lstsq(jac, -(dist - rho), rcond=None)
        c += step
        if float(np.linalg.norm(step)) < FIT_STEP_TOL_PX:
            break
    return c if bool(np.all(np.isfinite(c))) else None


def others_of(masks: BoolArray, t: int, index: int) -> BoolArray:
    """Union of every other track's mask on row ``t``."""
    found: BoolArray = np.any(np.delete(masks[t], index, axis=0), axis=0)
    return found


def track_radii(masks: BoolArray, index: int, params: Params) -> FloatArray:
    """Per row: median sqrt(A / pi) over the free rows within ``radius_window_rows``, else over
    every free row of the track, else NaN. A free row's mask touches no blocked pixel."""
    n_t = masks.shape[0]
    free_r = np.full(n_t, np.nan)
    for t in range(n_t):
        own = masks[t, index]
        block = blocked(others_of(masks, t, index), params.exclusion_px)
        if own.any() and not (own & block).any() and own.sum() >= params.area_min_px:
            free_r[t] = math.sqrt(float(own.sum()) / math.pi)
    whole = float(np.nanmedian(free_r)) if np.isfinite(free_r).any() else math.nan
    out = np.full(n_t, whole)
    w = params.radius_window_rows
    for t in range(n_t):
        near = free_r[max(0, t - w) : t + w + 1]
        if np.isfinite(near).any():
            out[t] = float(np.nanmedian(near))
    return out


def centre_of(
    mask: BoolArray, others: BoolArray, radius_px: float, params: Params
) -> tuple[FloatArray, float, CentreMethod]:
    """One row's centre, radius and method: circle of known radius on the silhouette, else the
    centroid, else absent below ``area_min_px``."""
    area = int(mask.sum())
    if area < params.area_min_px:
        return np.full(2, np.nan), math.nan, CentreMethod.ABSENT
    centroid = pixel_uv(mask).mean(axis=0)
    inset = params.boundary_inset_px
    if math.isfinite(radius_px) and radius_px > inset:
        contour = silhouette_contour(mask, blocked(others, params.exclusion_px))
        if (
            len(contour) >= params.contour_min_px
            and angular_span_deg(contour, centroid) >= params.min_arc_deg
        ):
            fitted = fit_known_radius(contour, radius_px - inset, centroid)
            if fitted is not None and float(np.linalg.norm(fitted - centroid)) <= radius_px:
                return fitted, radius_px, CentreMethod.CIRCLE
    fallback = radius_px if math.isfinite(radius_px) else math.sqrt(area / math.pi)
    return centroid, fallback, CentreMethod.CENTROID


class ContourCircleCentre:
    """Every row of a track of one of ``primitives`` gets this estimator; the others keep the
    lift's surface median."""

    name: ClassVar[str] = "contour_circle"
    version: ClassVar[str] = "1"
    Params: ClassVar[type[CentreParams]] = Params

    def __init__(self, params: CentreParams) -> None:
        if not isinstance(params, Params):
            raise TypeError(f"expected {Params.__name__}, got {type(params).__name__}")
        self.params = params

    def image_centres(
        self, masks: BoolArray, primitives: Sequence[PrimitiveKind]
    ) -> ImageCentres | None:
        n_t, n_obj = masks.shape[:2]
        uv = np.full((n_t, n_obj, 2), np.nan)
        radius = np.full((n_t, n_obj), np.nan)
        method = np.full((n_t, n_obj), CentreMethod.SURFACE_MEDIAN.value)
        for n, kind in enumerate(primitives):
            if kind not in self.params.primitives:
                continue
            radii = track_radii(masks, n, self.params)
            for t in range(n_t):
                found = centre_of(masks[t, n], others_of(masks, t, n), radii[t], self.params)
                uv[t, n], radius[t, n], method[t, n] = found[0], found[1], found[2].value
        return ImageCentres(uv=uv, method=method, radius_px=radius)


PLUGINS = (ContourCircleCentre,)
