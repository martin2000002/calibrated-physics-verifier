"""depth/metric: the depth model's focal and metric scale against the truth."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from metric_cases import depth_on, metric_case, native_sampling, sloped_z, squares_track
from numpy.typing import NDArray

from physverify.diagnosis.measurement_cases import MetricOutcome
from physverify.diagnosis.metrics.depth_metric import MetricDepthMetric
from physverify.schema.clip import FrameSampling
from physverify.schema.error_profile import FOCAL_REL, SCALE_REL
from physverify.schema.perception import DepthKind, DepthMap
from physverify.schema.stages import Stage

FRAMES, SIZE = 2, 20
TRUE_FOCAL_PX = 24.0
OBJECT_SCALE = 1.2
BACKGROUND_SCALE = 3.0


def sampling() -> FrameSampling:
    return native_sampling(FRAMES, SIZE, SIZE)


def true_z() -> NDArray[np.float64]:
    return sloped_z(FRAMES, SIZE)


def depth(
    values: NDArray[np.float64], kind: DepthKind = DepthKind.METRIC_Z, focal: float | None = None
) -> DepthMap:
    return dataclasses.replace(depth_on(values, sampling(), kind), focal_px=focal)


def metric_truth() -> DepthMap:
    return depth(true_z(), focal=TRUE_FOCAL_PX)


def scaled_on_objects() -> NDArray[np.float64]:
    """The truth times OBJECT_SCALE on the object pixels and BACKGROUND_SCALE elsewhere."""
    objects = squares_track(FRAMES, SIZE).masks.any(axis=1)
    return np.where(objects, OBJECT_SCALE, BACKGROUND_SCALE) * true_z()


class MetricStore:
    """The one ground-truth call the metric makes: the metric z layer."""

    def depth(self, sampling: FrameSampling) -> DepthMap:
        assert sampling == metric_truth().sampling
        return metric_truth()


def measure(
    pred: DepthMap, truth: DepthMap, *, track: bool = True, gt: Any = None
) -> MetricOutcome:
    context = {Stage.TRACK: squares_track(FRAMES, SIZE)} if track else {}
    case = dataclasses.replace(metric_case(pred, truth, context=context), gt=gt)
    return MetricDepthMetric(case.options).samples(case)


def test_the_focal_error_is_relative_to_the_true_focal() -> None:
    outcome = measure(depth(true_z(), focal=0.9 * TRUE_FOCAL_PX), metric_truth())
    np.testing.assert_allclose(outcome.samples[FOCAL_REL], [-0.1])


def test_a_depth_without_focal_has_no_focal_error() -> None:
    assert outcome_size(measure(depth(true_z()), metric_truth()), FOCAL_REL) == 0


def test_the_scale_error_is_the_median_ratio_on_object_pixels() -> None:
    outcome = measure(depth(scaled_on_objects()), metric_truth())
    np.testing.assert_allclose(outcome.samples[SCALE_REL], [OBJECT_SCALE - 1.0], rtol=1e-6)


def test_a_relative_truth_is_replaced_by_the_stores_metric_depth() -> None:
    relative = depth(true_z() / 5.0, DepthKind.RELATIVE_DEPTH, TRUE_FOCAL_PX)
    outcome = measure(depth(scaled_on_objects()), relative, gt=MetricStore())
    np.testing.assert_allclose(outcome.samples[SCALE_REL], [OBJECT_SCALE - 1.0], rtol=1e-6)


@pytest.mark.parametrize(
    ("pred_kind", "truth_kind", "track"),
    [
        (DepthKind.RELATIVE_DEPTH, DepthKind.METRIC_Z, True),
        (DepthKind.METRIC_Z, DepthKind.METRIC_Z, False),
        (DepthKind.METRIC_Z, DepthKind.RELATIVE_DEPTH, True),
    ],
)
def test_no_scale_error_without_a_metric_prediction_objects_or_metric_truth(
    pred_kind: DepthKind, truth_kind: DepthKind, track: bool
) -> None:
    outcome = measure(depth(true_z(), pred_kind), depth(true_z(), truth_kind), track=track)
    assert outcome_size(outcome, SCALE_REL) == 0


def outcome_size(outcome: MetricOutcome, quantity: str) -> int:
    return int(outcome.samples[quantity].size)
