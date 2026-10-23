"""depth/object_rel: per-object depth error after one scale per clip."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest
from metric_cases import METRIC_OPTIONS as OPTIONS
from metric_cases import (
    depth_on,
    metric_case,
    native_sampling,
    sloped_z,
    squares_track,
    two_squares,
)
from numpy.typing import NDArray
from synthetic_clip import manifest

from physverify.diagnosis.measurement_cases import MetricOutcome
from physverify.diagnosis.metrics.depth_object import ObjectDepthMetric, relative_object_error
from physverify.schema.clip import FrameSampling
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import (
    CameraModel,
    DepthKind,
    DepthMap,
    IntrinsicsSource,
    MaskTrack,
)
from physverify.schema.stages import Stage

FRAMES, SIZE = 2, 20
FLOAT32_TOL = 1e-6


def sampling() -> FrameSampling:
    return native_sampling(FRAMES, SIZE, SIZE)


def object_masks() -> NDArray[np.bool_]:
    return two_squares(FRAMES, SIZE)


def masks_track() -> MaskTrack:
    return squares_track(FRAMES, SIZE)


def true_z() -> NDArray[np.float64]:
    return sloped_z(FRAMES, SIZE)


def depth_map(values: NDArray[np.float64], kind: DepthKind = DepthKind.METRIC_Z) -> DepthMap:
    return depth_on(values, sampling(), kind)


def with_ball_error(c: float, eps: float) -> NDArray[np.float64]:
    factor = np.full(true_z().shape, c)
    factor[object_masks()[:, 0]] *= 1.0 + eps
    return factor * true_z()


def measure(pred: DepthMap, context: Mapping[Stage, StageOutput]) -> MetricOutcome:
    case = metric_case(pred, depth_map(true_z()), context=context)
    return ObjectDepthMetric(OPTIONS).samples(case)


def test_global_scale_leaves_no_object_error() -> None:
    outcome = measure(depth_map(1.25 * true_z()), {Stage.TRACK: masks_track()})
    residual = outcome.samples["depth_rel"]
    assert residual.size == FRAMES * 2
    np.testing.assert_allclose(residual, np.zeros(FRAMES * 2), atol=FLOAT32_TOL)


def test_relative_prediction_is_scale_free_too() -> None:
    pred = depth_map(1.25 * true_z(), DepthKind.RELATIVE_DEPTH)
    outcome = measure(pred, {Stage.TRACK: masks_track()})
    np.testing.assert_allclose(outcome.samples["depth_rel"], np.zeros(FRAMES * 2), atol=FLOAT32_TOL)


def test_one_object_off_by_eps_keeps_the_ratio_between_objects() -> None:
    eps = 0.1
    outcome = measure(depth_map(with_ball_error(1.25, eps)), {Stage.TRACK: masks_track()})
    residual = outcome.samples["depth_rel"].reshape(FRAMES, 2)
    np.testing.assert_allclose((1 + residual[:, 0]) / (1 + residual[:, 1]), 1 + eps, atol=1e-6)
    np.testing.assert_allclose(residual[:, 0], math.sqrt(1 + eps) - 1, atol=1e-6)
    np.testing.assert_allclose(residual[:, 1], 1 / math.sqrt(1 + eps) - 1, atol=1e-6)


def test_camera_context_does_not_change_the_depth_error() -> None:
    k = np.array([[30.0, 0.0, 10.0], [0.0, 30.0, 10.0], [0.0, 0.0, 1.0]])
    camera = CameraModel(
        sampling=sampling(),
        K=k,
        source=IntrinsicsSource.GROUND_TRUTH,
        fov_deg=36.9,
        fov_sigma_deg=0.0,
    )
    pred = depth_map(with_ball_error(1.25, 0.1))
    with_camera = measure(pred, {Stage.TRACK: masks_track(), Stage.CAMERA: camera})
    without = measure(pred, {Stage.TRACK: masks_track()})
    np.testing.assert_allclose(with_camera.samples["depth_rel"], without.samples["depth_rel"])


def test_without_track_context_the_error_is_nan() -> None:
    summary = ObjectDepthMetric(OPTIONS).compute(
        depth_map(true_z()), depth_map(true_z()), manifest()
    )
    assert summary["depth_rel.n"] == 0.0
    assert math.isnan(summary["depth_rel.mean"])


def test_relative_object_error_ignores_non_positive_pairs() -> None:
    z_true = np.array([[1.0, 2.0], [np.nan, 4.0]])
    z_pred = np.array([[2.0, 4.0], [1.0, 0.0]])
    np.testing.assert_allclose(relative_object_error(z_pred, z_true), [0.0, 0.0])
    assert relative_object_error(np.zeros((1, 1)), np.ones((1, 1))).size == 0


def test_wrong_output_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        measure(masks_track(), {})  # type: ignore[arg-type]
