"""DEPTH: per-object median depth of the LIFT conversion, after one scale per clip."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.cases import (
    BoolArray,
    FloatArray,
    MetricCase,
    MetricOutcome,
    bare_compute,
    stage_output,
)
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.plan import ResolvedPlan
from physverify.extraction.providers._lifting import SummaryOptions, lift_depth, object_summary
from physverify.extraction.providers.lift_backproject import Params as LiftParams
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import CameraModel, DepthMap, MaskTrack
from physverify.schema.stages import Stage


def lift_params(plan: ResolvedPlan | None) -> LiftParams:
    """The plan's back-projection params, so the depth is read exactly as LIFT reads it."""
    if plan is not None and Stage.LIFT in plan.names:
        params = plan.step(Stage.LIFT).params
        if isinstance(params, LiftParams):
            return params
    return LiftParams()


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


@dataclass(frozen=True)
class ObjectDepthMetric:
    """``depth_rel`` per object and frame. Needs context TRACK (masks); CAMERA is optional,
    since the median depth does not depend on K. Without TRACK ``depth_rel`` is NaN."""

    stage: ClassVar[Stage] = Stage.DEPTH
    name: ClassVar[str] = "object_rel"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, DepthMap) or not isinstance(truth, DepthMap):
            raise TypeError("object depth metric compares two DepthMap")
        track = stage_output(case.context, Stage.TRACK, MaskTrack)
        if track is None:
            return MetricOutcome(samples=MappingProxyType({"depth_rel": np.zeros(0)}))
        camera = stage_output(case.context, Stage.CAMERA, CameraModel)
        k = camera.K if camera is not None else np.eye(3)
        params = lift_params(case.plan)
        residual = relative_object_error(
            object_depths(pred, track, k, params), object_depths(truth, track, k, params)
        )
        return MetricOutcome(samples=MappingProxyType({"depth_rel": residual}))


PLUGINS = (ObjectDepthMetric,)
