"""DETECT: label accuracy over the box pairs matched by ``detect/box`` (ADR 0014 §2.1)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from physverify.diagnosis.measurement_cases import MetricCase, MetricOutcome, bare_compute
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.matching import DEFAULT_MIN_IOU, match_boxes
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import Detections
from physverify.schema.stages import Stage


def labels_by_id(detections: Detections) -> dict[str, str]:
    return {
        str(i): str(label).strip().lower()
        for i, label in zip(detections.object_ids.tolist(), detections.labels.tolist(), strict=True)
    }


@dataclass(frozen=True)
class LabelMetric:
    """``label_accuracy``: matched pairs whose labels agree (case and spaces ignored) over
    matched pairs. Unmatched objects count in ``detect/box`` recall, not here. No context."""

    stage: ClassVar[Stage] = Stage.DETECT
    name: ClassVar[str] = "label"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, Detections) or not isinstance(truth, Detections):
            raise TypeError("label metric compares two Detections")
        pairs = match_boxes(pred, truth, DEFAULT_MIN_IOU).pairs
        predicted, expected = labels_by_id(pred), labels_by_id(truth)
        agree = sum(predicted[p] == expected[g] for p, g, _ in pairs)
        return MetricOutcome(
            samples=MappingProxyType({}),
            ratios=MappingProxyType({"label_accuracy": (float(agree), float(len(pairs)))}),
        )


PLUGINS = (LabelMetric,)
