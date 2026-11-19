"""detect/label: label agreement on the box pairs matched by detect/box."""

from __future__ import annotations

import numpy as np
import pytest
from metric_cases import METRIC_OPTIONS as OPTIONS
from metric_cases import metric_case, native_sampling
from synthetic_clip import manifest

from physverify.diagnosis.measurement_cases import MetricOutcome
from physverify.diagnosis.metrics.detect_label import LabelMetric
from physverify.schema.clip import FrameSampling
from physverify.schema.perception import Detections

BOX_A = (10.0, 10.0, 20.0, 20.0)
BOX_B = (30.0, 30.0, 40.0, 44.0)
FAR = (50.0, 30.0, 60.0, 40.0)


def sampling() -> FrameSampling:
    return native_sampling(1, 64, 48)


def detections(boxes: list[tuple[float, float, float, float]], labels: list[str]) -> Detections:
    return Detections(
        sampling=sampling(),
        frame_index=0,
        boxes_xyxy=np.array(boxes, dtype=np.float64).reshape(-1, 4),
        scores=np.ones(len(labels)),
        labels=np.array(labels, dtype=str),
        object_ids=np.array([f"o{i}" for i in range(len(labels))], dtype=str),
    )


def measure(pred: Detections, truth: Detections) -> MetricOutcome:
    case = metric_case(pred, truth)
    return LabelMetric(OPTIONS).samples(case)


def test_same_labels_ignoring_case_and_spaces_agree() -> None:
    outcome = measure(
        detections([BOX_A, BOX_B], [" Cube", "sphere "]),
        detections([BOX_A, BOX_B], ["cube", "sphere"]),
    )
    assert outcome.ratios["label_accuracy"] == (2.0, 2.0)


def test_swapped_labels_disagree() -> None:
    outcome = measure(
        detections([BOX_A, BOX_B], ["sphere", "cube"]),
        detections([BOX_A, BOX_B], ["cube", "sphere"]),
    )
    assert outcome.ratios["label_accuracy"] == (0.0, 2.0)


def test_unmatched_objects_are_left_to_recall() -> None:
    outcome = measure(
        detections([BOX_A, FAR], ["cube", "cube"]), detections([BOX_A, BOX_B], ["cube", "sphere"])
    )
    assert outcome.ratios["label_accuracy"] == (1.0, 1.0)
    assert dict(outcome.samples) == {}


def test_compute_reports_the_rate() -> None:
    truth = detections([BOX_A, BOX_B], ["cube", "sphere"])
    summary = LabelMetric(OPTIONS).compute(
        detections([BOX_A, BOX_B], ["cube", "cube"]), truth, manifest()
    )
    assert summary["label_accuracy"] == pytest.approx(0.5)


def test_wrong_output_type_is_rejected() -> None:
    truth = detections([BOX_A], ["cube"])
    with pytest.raises(TypeError):
        LabelMetric(OPTIONS).compute(sampling(), truth, manifest())  # type: ignore[arg-type]
