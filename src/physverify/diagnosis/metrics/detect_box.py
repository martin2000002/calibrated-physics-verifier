"""DETECT: per-object box IoU after Hungarian matching, with set recall and precision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.cases import MetricCase, MetricOutcome, bare_compute
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.matching import DEFAULT_MIN_IOU, match_boxes
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import Detections
from physverify.schema.stages import Stage


@dataclass(frozen=True)
class BoxMetric:
    """``box_iou`` per ground-truth object (0 when unmatched). Needs no context."""

    stage: ClassVar[Stage] = Stage.DETECT
    name: ClassVar[str] = "box"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, Detections) or not isinstance(truth, Detections):
            raise TypeError("box metric compares two Detections")
        ids = match_boxes(pred, truth, DEFAULT_MIN_IOU)
        matched = {gt_id: score for _, gt_id, score in ids.pairs}
        ious = np.array([matched.get(g, 0.0) for g in truth.object_ids.tolist()], dtype=np.float64)
        n_matched = float(len(ids.pairs))
        return MetricOutcome(
            samples=MappingProxyType({"box_iou": ious}),
            ratios=MappingProxyType(
                {
                    "recall": (n_matched, float(truth.object_ids.size)),
                    "precision": (n_matched, float(pred.object_ids.size)),
                }
            ),
        )


PLUGINS = (BoxMetric,)
