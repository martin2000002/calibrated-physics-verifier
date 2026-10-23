"""DEPTH: per-frame AbsRel and delta1 after aligning the prediction to the truth by its kind."""

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
)
from physverify.diagnosis.metrics._depth import aligned_depth, valid_pixels
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import DepthMap
from physverify.schema.stages import Stage

DELTA1 = 1.25


@dataclass(frozen=True)
class AbsRelMetric:
    """``depth_absrel`` per frame with valid pixels; ``delta1`` pooled over pixels. No context."""

    stage: ClassVar[Stage] = Stage.DEPTH
    name: ClassVar[str] = "absrel"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, DepthMap) or not isinstance(truth, DepthMap):
            raise TypeError("absrel metric compares two DepthMap")
        valid = valid_pixels(pred, truth)
        z_hat = aligned_depth(pred, truth, valid, case)
        z = truth.depth.astype(np.float64)
        absrel = [
            float(np.mean(np.abs(z_hat[t][valid[t]] - z[t][valid[t]]) / z[t][valid[t]]))
            for t in range(z.shape[0])
            if valid[t].any()
        ]
        ratio = np.maximum(z_hat[valid] / z[valid], z[valid] / z_hat[valid])
        return MetricOutcome(
            samples=MappingProxyType({"depth_absrel": np.array(absrel, dtype=np.float64)}),
            ratios=MappingProxyType(
                {"delta1": (float(np.count_nonzero(ratio < DELTA1)), float(ratio.size))}
            ),
        )


PLUGINS = (AbsRelMetric,)
