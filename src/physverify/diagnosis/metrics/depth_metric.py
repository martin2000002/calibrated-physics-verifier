"""DEPTH: the focal and the metric scale a depth model estimates, against the true camera and the
true metric depth. Both errors are the depth model's, so they are attributed to this stage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.measurement_cases import (
    FloatArray,
    MetricCase,
    MetricOutcome,
    bare_compute,
    stage_output,
)
from physverify.diagnosis.metrics._depth import valid_pixels
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.clip import ClipManifest
from physverify.schema.error_profile import FOCAL_REL, SCALE_REL
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import DepthKind, DepthMap, MaskTrack
from physverify.schema.stages import Stage


def focal_error(pred: DepthMap, truth: DepthMap) -> FloatArray:
    """f_pred / f_true - 1; empty unless both depths carry a focal."""
    if pred.focal_px is None or truth.focal_px is None:
        return np.zeros(0)
    return np.array([pred.focal_px / truth.focal_px - 1.0])


def metric_truth(case: MetricCase, truth: DepthMap) -> DepthMap | None:
    """The truth itself when it is metric, else the ground-truth store's metric z."""
    if truth.kind is DepthKind.METRIC_Z:
        return truth
    return None if case.gt is None else case.gt.depth(truth.sampling)


def scale_error(pred: DepthMap, truth: DepthMap | None, track: MaskTrack | None) -> FloatArray:
    """median(z_pred / z_true) - 1 over the object pixels of the clip; empty unless the
    prediction is metric z and there are a metric truth and object pixels."""
    if pred.kind is not DepthKind.METRIC_Z or truth is None or track is None:
        return np.zeros(0)
    pixels = valid_pixels(pred, truth) & track.masks.any(axis=1)
    if not pixels.any():
        return np.zeros(0)
    ratio = pred.depth[pixels].astype(np.float64) / truth.depth[pixels].astype(np.float64)
    return np.array([float(np.median(ratio)) - 1.0])


@dataclass(frozen=True)
class MetricDepthMetric:
    """One ``focal_rel`` and one ``scale_rel`` sample per clip. The scale needs context TRACK
    and a metric truth (the case's own, or the ground-truth store's when the truth depth is
    relative); the focal needs none."""

    stage: ClassVar[Stage] = Stage.DEPTH
    name: ClassVar[str] = "metric"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, DepthMap) or not isinstance(truth, DepthMap):
            raise TypeError("metric depth metric compares two DepthMap")
        track = stage_output(case.context, Stage.TRACK, MaskTrack)
        samples = {
            FOCAL_REL: focal_error(pred, truth),
            SCALE_REL: scale_error(pred, metric_truth(case, truth), track),
        }
        return MetricOutcome(samples=MappingProxyType(samples))


PLUGINS = (MetricDepthMetric,)
