"""depth/absrel: alignment by depth kind and per-frame AbsRel."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest
from numpy.typing import NDArray
from synthetic_clip import manifest

from physverify.diagnosis.measurement_cases import MetricCase, MetricOutcome
from physverify.diagnosis.metrics.depth_absrel import AbsRelMetric
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.clip import FrameSampling
from physverify.schema.perception import DepthKind, DepthMap

OPTIONS = MetricSettings(
    boundary_tol_frac=0.008,
    depth_alignment="clip",
    depth_eps_frac=1e-3,
    contact_min_tiou=0.3,
    impact_half_window_s=0.125,
)
FRAMES, HEIGHT, WIDTH = 2, 6, 8
FLOAT32_TOL = 1e-6


def sampling() -> FrameSampling:
    return FrameSampling(
        frame_indices=np.arange(FRAMES, dtype=np.int64),
        width=WIDTH,
        height=HEIGHT,
        native_width=WIDTH,
        native_height=HEIGHT,
        seconds_per_frame=1.0 / 24.0,
        step=1,
        max_side=None,
    )


def true_z() -> NDArray[np.float64]:
    t, rows, cols = np.mgrid[:FRAMES, :HEIGHT, :WIDTH]
    z: NDArray[np.float64] = 1.0 + 0.25 * cols + 0.1 * rows + 0.5 * t
    return z


def depth_map(
    values: NDArray[np.float64], kind: DepthKind, valid: NDArray[np.bool_] | None = None
) -> DepthMap:
    return DepthMap(
        sampling=sampling(),
        depth=values.astype(np.float32),
        kind=kind,
        valid=np.ones(values.shape, dtype=bool) if valid is None else valid,
    )


def measure(pred: DepthMap, truth: DepthMap, options: MetricSettings = OPTIONS) -> MetricOutcome:
    case = MetricCase(
        clip=manifest(),
        pred=pred,
        truth=truth,
        context=MappingProxyType({}),
        gt=None,
        options=options,
        plan=None,
    )
    return AbsRelMetric(options).samples(case)


def metric_truth() -> DepthMap:
    return depth_map(true_z(), DepthKind.METRIC_Z)


def test_affine_disparity_is_aligned_exactly() -> None:
    a, b = 2.0, 0.5
    outcome = measure(depth_map(a / true_z() + b, DepthKind.AFFINE_DISPARITY), metric_truth())
    np.testing.assert_allclose(outcome.samples["depth_absrel"], np.zeros(FRAMES), atol=FLOAT32_TOL)
    assert outcome.ratios["delta1"] == (float(true_z().size), float(true_z().size))


def test_relative_depth_is_aligned_by_the_median_ratio() -> None:
    outcome = measure(depth_map(3.0 * true_z(), DepthKind.RELATIVE_DEPTH), metric_truth())
    np.testing.assert_allclose(outcome.samples["depth_absrel"], np.zeros(FRAMES), atol=FLOAT32_TOL)


def test_metric_depth_against_metric_truth_is_not_rescaled() -> None:
    outcome = measure(depth_map(1.1 * true_z(), DepthKind.METRIC_Z), metric_truth())
    np.testing.assert_allclose(outcome.samples["depth_absrel"], np.full(FRAMES, 0.1), atol=1e-6)


def test_metric_depth_against_relative_truth_is_rescaled() -> None:
    truth = depth_map(true_z(), DepthKind.RELATIVE_DEPTH)
    outcome = measure(depth_map(1.1 * true_z(), DepthKind.METRIC_Z), truth)
    np.testing.assert_allclose(outcome.samples["depth_absrel"], np.zeros(FRAMES), atol=FLOAT32_TOL)


def test_frame_alignment_absorbs_a_per_frame_scale() -> None:
    factor = np.array([2.0, 5.0])[:, None, None]
    pred = depth_map(factor * true_z(), DepthKind.RELATIVE_DEPTH)
    per_frame = measure(
        pred, metric_truth(), OPTIONS.model_copy(update={"depth_alignment": "frame"})
    )
    per_clip = measure(pred, metric_truth())
    np.testing.assert_allclose(per_frame.samples["depth_absrel"], np.zeros(FRAMES), atol=1e-6)
    assert per_clip.samples["depth_absrel"].max() > 0.1


def test_frames_without_valid_pixels_give_no_sample() -> None:
    valid = np.ones((FRAMES, HEIGHT, WIDTH), dtype=bool)
    valid[1] = False
    pred = depth_map(true_z(), DepthKind.METRIC_Z, valid)
    outcome = measure(pred, metric_truth())
    assert outcome.samples["depth_absrel"].size == 1
    assert outcome.ratios["delta1"][1] == float(HEIGHT * WIDTH)


def test_disparity_truth_is_rejected() -> None:
    truth = depth_map(1.0 / true_z(), DepthKind.AFFINE_DISPARITY)
    with pytest.raises(ValueError, match="not a disparity"):
        measure(metric_truth(), truth)


def test_compute_summarizes_without_context() -> None:
    summary = AbsRelMetric(OPTIONS).compute(metric_truth(), metric_truth(), manifest())
    assert summary["depth_absrel.mean"] == pytest.approx(0.0, abs=FLOAT32_TOL)
    assert summary["delta1"] == 1.0
