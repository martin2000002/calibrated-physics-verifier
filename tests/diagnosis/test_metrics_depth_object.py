"""depth/object_rel: per-object depth error after one scale per clip (ADR 0007 §7)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pytest
from numpy.typing import NDArray
from synthetic_clip import manifest

from physverify.diagnosis.cases import MetricCase, MetricOutcome
from physverify.diagnosis.metrics.depth_object import ObjectDepthMetric, relative_object_error
from physverify.diagnosis.settings import MetricSettings
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

OPTIONS = MetricSettings(
    boundary_tol_frac=0.008,
    depth_alignment="clip",
    depth_eps_frac=1e-3,
    contact_min_tiou=0.3,
    impact_half_window_s=0.125,
)
FRAMES, SIZE = 2, 20
FLOAT32_TOL = 1e-6


def sampling() -> FrameSampling:
    return FrameSampling(
        frame_indices=np.arange(FRAMES, dtype=np.int64),
        width=SIZE,
        height=SIZE,
        native_width=SIZE,
        native_height=SIZE,
        seconds_per_frame=1.0 / 24.0,
        step=1,
        max_side=None,
    )


def object_masks() -> NDArray[np.bool_]:
    """(T, 2, h, w): "ball" a 7x7 square, "box" an 8x8 square, disjoint."""
    masks = np.zeros((FRAMES, 2, SIZE, SIZE), dtype=bool)
    masks[:, 0, 2:9, 2:9] = True
    masks[:, 1, 10:18, 10:18] = True
    return masks


def masks_track() -> MaskTrack:
    return MaskTrack(
        sampling=sampling(),
        object_ids=np.array(["ball", "box"]),
        labels=np.array(["ball", "box"]),
        masks=object_masks(),
        scores=np.ones((FRAMES, 2)),
    )


def true_z() -> NDArray[np.float64]:
    t, rows, cols = np.mgrid[:FRAMES, :SIZE, :SIZE]
    z: NDArray[np.float64] = 2.0 + 0.1 * cols + 0.05 * rows + 0.3 * t
    return z


def depth_map(values: NDArray[np.float64], kind: DepthKind = DepthKind.METRIC_Z) -> DepthMap:
    return DepthMap(
        sampling=sampling(),
        depth=values.astype(np.float32),
        kind=kind,
        valid=np.ones(values.shape, dtype=bool),
    )


def with_ball_error(c: float, eps: float) -> NDArray[np.float64]:
    factor = np.full(true_z().shape, c)
    factor[object_masks()[:, 0]] *= 1.0 + eps
    return factor * true_z()


def measure(pred: DepthMap, context: Mapping[Stage, StageOutput]) -> MetricOutcome:
    case = MetricCase(
        clip=manifest(),
        pred=pred,
        truth=depth_map(true_z()),
        context=MappingProxyType(dict(context)),
        gt=None,
        options=OPTIONS,
        plan=None,
    )
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
