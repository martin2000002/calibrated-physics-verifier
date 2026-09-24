"""SCALE: relative error of meters-per-unit against the conditional ground-truth fit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.cases import FloatArray, MetricCase, MetricOutcome, bare_compute
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.scale import ScaleEstimate
from physverify.schema.stages import Stage

SCALE_REL = "scale_rel"


def source_key(estimate: ScaleEstimate) -> str:
    """Sample name of ``scale_rel`` labelled by the source that produced the estimate."""
    return f"{SCALE_REL}/{estimate.source.value}"


@dataclass(frozen=True)
class ScaleRelativeMetric:
    """``scale_rel = s_pred/s_true - 1``, also under ``scale_rel/<source>``. Needs no context."""

    stage: ClassVar[Stage] = Stage.SCALE
    name: ClassVar[str] = "relative"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, ScaleEstimate) or not isinstance(truth, ScaleEstimate):
            raise TypeError("scale metric compares two ScaleEstimate")
        s_pred, s_true = pred.meters_per_unit, truth.meters_per_unit
        residual: FloatArray = np.zeros(0)
        if s_pred is not None and s_true is not None:
            residual = np.array([s_pred / s_true - 1.0])
        return MetricOutcome(
            samples=MappingProxyType({SCALE_REL: residual, source_key(pred): residual}),
            ratios=MappingProxyType({"coverage": (float(s_pred is not None), 1.0)}),
        )


PLUGINS = (ScaleRelativeMetric,)
