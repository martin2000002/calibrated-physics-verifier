"""DEPTH: the per-object depth residual of ``object_rel`` split into its parts."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.measurement_cases import (
    BoolArray,
    FloatArray,
    MetricCase,
    MetricOutcome,
    bare_compute,
    stage_output,
)
from physverify.diagnosis.metrics._depth import (
    aligned_depth,
    comparable_pairs,
    object_depths,
    relative_object_error,
    valid_pixels,
)
from physverify.diagnosis.metrics._lift import lift_params
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.backprojection import DISPARITY_EPS, ShiftStrategy
from physverify.extraction.lift_params import LiftParams
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import CameraModel, DepthKind, DepthMap, MaskTrack
from physverify.schema.stages import Stage

QUANTITIES = (
    "depth_rel_between",
    "depth_rel_within",
    "depth_d2_rel",
    "depth_rel_oracle_shift",
    "depth_absrel_objects",
)


def residual_matrix(z_pred: FloatArray, z_true: FloatArray) -> FloatArray:
    """(T, N) residual of ``relative_object_error``; NaN where that metric drops the pair."""
    ok = comparable_pairs(z_pred, z_true)
    out = np.full(z_true.shape, np.nan)
    out[ok] = relative_object_error(z_pred, z_true)
    return out


def between_within(residual: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Per-object mean residual and the deviation from it, one value per finite object-frame.

    The mean is repeated on each of the object's frames so rms(r)^2 = rms(between)^2 +
    rms(within)^2 holds exactly over the same samples as ``object_rel``."""
    ok = np.isfinite(residual)
    counts = ok.sum(axis=0)
    sums = np.where(ok, residual, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    between = np.broadcast_to(means, residual.shape)[ok]
    within = (residual - means)[ok]
    return between, within


def second_difference_error(residual: FloatArray, z_true: FloatArray) -> FloatArray:
    """(dd(s z_pred) - dd(z_true)) / mean_t z_true on interior frames with three finite depths,
    dd the second difference in time."""
    if residual.shape[0] < 3:
        return np.zeros(0)
    z_pred = (residual + 1.0) * z_true
    d2_pred = z_pred[2:] - 2.0 * z_pred[1:-1] + z_pred[:-2]
    d2_true = z_true[2:] - 2.0 * z_true[1:-1] + z_true[:-2]
    usable = np.where(np.isfinite(residual), z_true, np.nan)
    finite = np.isfinite(usable).any(axis=0)
    scale = np.full(z_true.shape[1], np.nan)
    scale[finite] = np.nanmean(usable[:, finite], axis=0)
    error = (d2_pred - d2_true) / scale
    values: FloatArray = error[np.isfinite(error)]
    return values


def oracle_shifted(depth: DepthMap, truth: DepthMap, objects: BoolArray) -> DepthMap | None:
    """``depth`` plus the offset b/a of one fit a d + b ~ target per clip, pooled over every
    object pixel of every frame (one shift for the clip, not one per object).

    The target is 1/z for affine disparity and z otherwise, so the shifted map has the zero
    offset LIFT's ``zero_shift`` assumes. It uses the truth: an oracle bound, not a measurement.
    None when the fit is undefined: too few pixels, a <= 0 (depth order inverted) or no object
    pixel left positive after the shift."""
    d = depth.depth.astype(np.float64)
    z = truth.depth.astype(np.float64)
    pixels = objects & valid_pixels(depth, truth)
    if np.count_nonzero(pixels) < 2:
        return None
    target = 1.0 / z[pixels] if depth.kind is DepthKind.AFFINE_DISPARITY else z[pixels]
    design = np.column_stack([d[pixels], np.ones(np.count_nonzero(pixels))])
    (a, b), *_ = np.linalg.lstsq(design, target, rcond=None)
    if not (np.isfinite(a) and np.isfinite(b) and a > np.finfo(np.float64).eps):
        return None
    shifted = d + b / a
    if not np.any(shifted[pixels] > DISPARITY_EPS):
        return None
    return dataclasses.replace(depth, depth=shifted.astype(depth.depth.dtype))


def oracle_shift_error(
    pred: DepthMap, truth: DepthMap, track: MaskTrack, k: FloatArray, params: LiftParams
) -> FloatArray:
    objects = track.masks.any(axis=1)
    zero = params.model_copy(update={"shift": ShiftStrategy.ZERO_SHIFT})
    shifted = oracle_shifted(pred, truth, objects)
    if shifted is None:
        return np.zeros(0)
    return relative_object_error(
        object_depths(shifted, track, k, zero), object_depths(truth, track, k, zero)
    )


def object_absrel(
    pred: DepthMap, truth: DepthMap, track: MaskTrack, case: MetricCase
) -> FloatArray:
    """Per-frame AbsRel on object pixels, aligned over all valid pixels exactly as ``absrel``."""
    valid = valid_pixels(pred, truth)
    z_hat = aligned_depth(pred, truth, valid, case)
    z = truth.depth.astype(np.float64)
    on_objects = valid & track.masks.any(axis=1)
    return np.array(
        [
            float(
                np.mean(np.abs(z_hat[t][on_objects[t]] - z[t][on_objects[t]]) / z[t][on_objects[t]])
            )
            for t in range(z.shape[0])
            if on_objects[t].any()
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class ObjectDepthPartsMetric:
    """Diagnostic split of ``depth/object_rel``: between/within objects, temporal second
    difference, an oracle-shift bound and AbsRel on object pixels. Needs context TRACK;
    CAMERA is optional. ``depth_rel_oracle_shift`` fits the shift against the truth."""

    stage: ClassVar[Stage] = Stage.DEPTH
    name: ClassVar[str] = "object_parts"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, DepthMap) or not isinstance(truth, DepthMap):
            raise TypeError("object depth parts metric compares two DepthMap")
        track = stage_output(case.context, Stage.TRACK, MaskTrack)
        if track is None:
            return MetricOutcome(samples=MappingProxyType({q: np.zeros(0) for q in QUANTITIES}))
        camera = stage_output(case.context, Stage.CAMERA, CameraModel)
        k = camera.K if camera is not None else np.eye(3)
        params = lift_params(case.plan)
        z_true = object_depths(truth, track, k, params)
        residual = residual_matrix(object_depths(pred, track, k, params), z_true)
        between, within = between_within(residual)
        samples = {
            "depth_rel_between": between,
            "depth_rel_within": within,
            "depth_d2_rel": second_difference_error(residual, z_true),
            "depth_rel_oracle_shift": oracle_shift_error(pred, truth, track, k, params),
            "depth_absrel_objects": object_absrel(pred, truth, track, case),
        }
        return MetricOutcome(samples=MappingProxyType(samples))


PLUGINS = (ObjectDepthPartsMetric,)
