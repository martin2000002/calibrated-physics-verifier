"""CAMERA: field-of-view and focal errors, and the z-score against the declared FOV sigma."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.measurement_cases import MetricCase, MetricOutcome, bare_compute
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import CameraModel
from physverify.schema.stages import Stage


@dataclass(frozen=True)
class FovMetric:
    """One sample per clip; ``fov_z`` only when the prediction declares a finite sigma > 0."""

    stage: ClassVar[Stage] = Stage.CAMERA
    name: ClassVar[str] = "fov"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, CameraModel) or not isinstance(truth, CameraModel):
            raise TypeError("fov metric compares two CameraModel")
        fov_err = float(pred.fov_deg - truth.fov_deg)
        sigma = pred.fov_sigma_deg
        fov_z = [fov_err / sigma] if math.isfinite(sigma) and sigma > 0 else []
        return MetricOutcome(
            samples=MappingProxyType(
                {
                    "fov_err_deg": np.array([fov_err]),
                    "focal_rel": np.array([pred.K[0, 0] / truth.K[0, 0] - 1.0]),
                    "fov_z": np.array(fov_z, dtype=np.float64),
                }
            )
        )


PLUGINS = (FovMetric,)
