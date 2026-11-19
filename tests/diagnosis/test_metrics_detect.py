"""detect/box: analytic IoU cases."""

from __future__ import annotations

import numpy as np
import pytest
from metric_cases import METRIC_OPTIONS as OPTIONS
from metric_cases import metric_case, native_sampling
from synthetic_clip import manifest

from physverify.diagnosis.measurement_cases import MetricOutcome
from physverify.diagnosis.metrics.detect_box import BoxMetric
from physverify.schema.clip import FrameSampling
from physverify.schema.perception import Detections

BOX_A = (10.0, 10.0, 20.0, 20.0)
BOX_B = (30.0, 30.0, 40.0, 44.0)


def sampling() -> FrameSampling:
    return native_sampling(1, 64, 48)


def detections(boxes: list[tuple[float, float, float, float]], ids: list[str]) -> Detections:
    return Detections(
        sampling=sampling(),
        frame_index=0,
        boxes_xyxy=np.array(boxes, dtype=np.float64).reshape(-1, 4),
        scores=np.ones(len(ids)),
        labels=np.array(ids, dtype=str),
        object_ids=np.array(ids, dtype=str),
    )


def measure(pred: Detections, truth: Detections) -> MetricOutcome:
    case = metric_case(pred, truth)
    return BoxMetric(OPTIONS).samples(case)


def test_identical_boxes_score_one() -> None:
    truth = detections([BOX_A, BOX_B], ["a", "b"])
    outcome = measure(detections([BOX_A, BOX_B], ["p0", "p1"]), truth)
    np.testing.assert_allclose(outcome.samples["box_iou"], [1.0, 1.0], atol=1e-12)
    assert outcome.ratios["recall"] == (2.0, 2.0)
    assert outcome.ratios["precision"] == (2.0, 2.0)


def test_box_shifted_half_its_width_scores_one_third() -> None:
    shifted = (15.0, 10.0, 25.0, 20.0)
    outcome = measure(detections([shifted], ["p0"]), detections([BOX_A], ["a"]))
    assert outcome.samples["box_iou"][0] == pytest.approx(1.0 / 3.0, abs=1e-9)


def test_unmatched_truth_scores_zero_and_lowers_recall() -> None:
    far = (50.0, 30.0, 60.0, 40.0)
    outcome = measure(detections([far], ["p0"]), detections([BOX_A, BOX_B], ["a", "b"]))
    np.testing.assert_array_equal(outcome.samples["box_iou"], [0.0, 0.0])
    assert outcome.ratios["recall"] == (0.0, 2.0)
    assert outcome.ratios["precision"] == (0.0, 1.0)


def test_extra_prediction_lowers_precision_only() -> None:
    far = (50.0, 30.0, 60.0, 40.0)
    outcome = measure(detections([BOX_A, far], ["p0", "p1"]), detections([BOX_A], ["a"]))
    np.testing.assert_allclose(outcome.samples["box_iou"], [1.0], atol=1e-12)
    assert outcome.ratios["recall"] == (1.0, 1.0)
    assert outcome.ratios["precision"] == (1.0, 2.0)


def test_compute_summarizes_without_context() -> None:
    truth = detections([BOX_A, BOX_B], ["a", "b"])
    summary = BoxMetric(OPTIONS).compute(truth, truth, manifest())
    assert summary["box_iou.mean"] == pytest.approx(1.0)
    assert summary["box_iou.n"] == 2.0
    assert summary["recall"] == 1.0


def test_wrong_output_type_is_rejected() -> None:
    truth = detections([BOX_A], ["a"])
    with pytest.raises(TypeError):
        BoxMetric(OPTIONS).compute(sampling(), truth, manifest())  # type: ignore[arg-type]
