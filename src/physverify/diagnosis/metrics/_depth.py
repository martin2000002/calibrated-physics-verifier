"""Depth alignment and per-object depths shared by the DEPTH metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from physverify.diagnosis.measurement_cases import BoolArray, FloatArray, MetricCase
from physverify.extraction.backprojection import SummaryOptions, lift_depth, object_summary
from physverify.extraction.lift_params import LiftParams
from physverify.schema.perception import DepthKind, DepthMap, MaskTrack


def valid_pixels(pred: DepthMap, truth: DepthMap) -> BoolArray:
    d = pred.depth.astype(np.float64)
    z = truth.depth.astype(np.float64)
    valid: BoolArray = pred.valid & truth.valid & np.isfinite(d) & np.isfinite(z) & (z > 0)
    if pred.kind is not DepthKind.AFFINE_DISPARITY:
        valid &= d > 0
    return valid


def align(
    d: FloatArray, z: FloatArray, kind: DepthKind, eps_frac: float, metric_truth: bool
) -> FloatArray:
    """z_hat for matching 1-D pixel samples ``d`` (prediction) and ``z`` (truth depth).

    Affine disparity: least squares a d + b ~ 1/z, z_hat = 1/max(a d + b, eps) with
    eps = eps_frac * median(1/z). Relative depth (or metric depth against a relative truth):
    z_hat = median(z/d) d."""
    if kind is DepthKind.AFFINE_DISPARITY:
        inverse = 1.0 / z
        design = np.column_stack([d, np.ones_like(d)])
        (a, b), *_ = np.linalg.lstsq(design, inverse, rcond=None)
        eps = eps_frac * float(np.median(inverse))
        aligned: FloatArray = 1.0 / np.maximum(a * d + b, eps)
        return aligned
    if kind is DepthKind.METRIC_Z and metric_truth:
        return d
    scaled: FloatArray = float(np.median(z / d)) * d
    return scaled


def alignment_groups(n_frames: int, mode: str) -> Sequence[Sequence[int]]:
    return [list(range(n_frames))] if mode == "clip" else [[t] for t in range(n_frames)]


def aligned_depth(
    pred: DepthMap, truth: DepthMap, valid: BoolArray, case: MetricCase
) -> FloatArray:
    """z_hat on every valid pixel, NaN elsewhere."""
    if truth.kind is DepthKind.AFFINE_DISPARITY:
        raise ValueError("truth depth must be a depth, not a disparity")
    d, z = pred.depth.astype(np.float64), truth.depth.astype(np.float64)
    out = np.full(d.shape, np.nan)
    metric_truth = truth.kind is DepthKind.METRIC_Z
    for frames in alignment_groups(d.shape[0], case.options.depth_alignment):
        group = np.zeros_like(valid)
        group[frames] = valid[frames]
        if group.any():
            out[group] = align(
                d[group], z[group], pred.kind, case.options.depth_eps_frac, metric_truth
            )
    return out


def object_depths(
    depth: DepthMap, track: MaskTrack, k: FloatArray, params: LiftParams
) -> FloatArray:
    """(T, N) median depth of each object's surface; NaN where LIFT would give no summary."""
    z, valid, _ = lift_depth(depth, track, params.shift)
    options = SummaryOptions(params.erode_px, params.extent_percentiles, params.min_area_px)
    n_t, n_obj = track.masks.shape[:2]
    out = np.full((n_t, n_obj), np.nan)
    for t in range(n_t):
        for n in range(n_obj):
            summary = object_summary(track.masks[t, n], z[t], valid[t], k, options)
            if summary is not None:
                out[t, n] = summary.centre[2]
    return out


def comparable_pairs(z_pred: FloatArray, z_true: FloatArray) -> BoolArray:
    """The object-frames ``relative_object_error`` keeps: both depths finite and positive."""
    ok: BoolArray = np.isfinite(z_pred) & np.isfinite(z_true) & (z_pred > 0) & (z_true > 0)
    return ok


def relative_object_error(z_pred: FloatArray, z_true: FloatArray) -> FloatArray:
    """s z_pred/z_true - 1 with s = exp(median log(z_true/z_pred)) over the finite pairs."""
    ok = comparable_pairs(z_pred, z_true)
    if not ok.any():
        return np.zeros(0)
    scale = float(np.exp(np.median(np.log(z_true[ok] / z_pred[ok]))))
    residual: FloatArray = scale * z_pred[ok] / z_true[ok] - 1.0
    return residual
