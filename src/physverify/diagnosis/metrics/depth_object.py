"""DEPTH: per-object median depth of the LIFT conversion, after one scale per clip."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.measurement_cases import (
    MetricCase,
    MetricOutcome,
    bare_compute,
    stage_output,
)
from physverify.diagnosis.metrics._depth import (
    object_depths,
    relative_object_error,
)
from physverify.diagnosis.metrics._lift import lift_params
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import CameraModel, DepthMap, MaskTrack
from physverify.schema.stages import Stage


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
