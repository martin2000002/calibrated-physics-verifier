"""camera/fov: field-of-view, focal and z-score cases."""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
import pytest
from synthetic_clip import manifest

from physverify.diagnosis.measurement_cases import MetricCase, MetricOutcome
from physverify.diagnosis.metrics.camera_fov import FovMetric
from physverify.diagnosis.settings import MetricSettings
from physverify.schema.clip import FrameSampling
from physverify.schema.perception import CameraModel, IntrinsicsSource

OPTIONS = MetricSettings(
    boundary_tol_frac=0.008,
    depth_alignment="clip",
    depth_eps_frac=1e-3,
    contact_min_tiou=0.3,
    impact_half_window_s=0.125,
)
WIDTH, HEIGHT = 96, 72


def sampling() -> FrameSampling:
    return FrameSampling(
        frame_indices=np.arange(4, dtype=np.int64),
        width=WIDTH,
        height=HEIGHT,
        native_width=WIDTH,
        native_height=HEIGHT,
        seconds_per_frame=1.0 / 24.0,
        step=1,
        max_side=None,
    )


def camera(focal: float, fov_deg: float, sigma: float, source: IntrinsicsSource) -> CameraModel:
    k = np.array([[focal, 0.0, WIDTH / 2], [0.0, focal, HEIGHT / 2], [0.0, 0.0, 1.0]])
    return CameraModel(
        sampling=sampling(), K=k, source=source, fov_deg=fov_deg, fov_sigma_deg=sigma
    )


def truth() -> CameraModel:
    return camera(80.0, 60.0, 0.0, IntrinsicsSource.GROUND_TRUTH)


def measure(pred: CameraModel) -> MetricOutcome:
    case = MetricCase(
        clip=manifest(),
        pred=pred,
        truth=truth(),
        context=MappingProxyType({}),
        gt=None,
        options=OPTIONS,
        plan=None,
    )
    return FovMetric(OPTIONS).samples(case)


def test_identical_camera_has_no_error() -> None:
    outcome = measure(camera(80.0, 60.0, 2.0, IntrinsicsSource.ESTIMATED))
    np.testing.assert_array_equal(outcome.samples["fov_err_deg"], [0.0])
    np.testing.assert_array_equal(outcome.samples["focal_rel"], [0.0])
    np.testing.assert_array_equal(outcome.samples["fov_z"], [0.0])


def test_fov_offset_and_its_z_score() -> None:
    outcome = measure(camera(80.0, 63.0, 1.5, IntrinsicsSource.ASSUMED_FOV))
    assert outcome.samples["fov_err_deg"][0] == pytest.approx(3.0, abs=1e-9)
    assert outcome.samples["fov_z"][0] == pytest.approx(2.0, abs=1e-9)


def test_focal_ratio() -> None:
    outcome = measure(camera(96.0, 60.0, 1.0, IntrinsicsSource.ESTIMATED))
    assert outcome.samples["focal_rel"][0] == pytest.approx(0.2, abs=1e-9)


@pytest.mark.parametrize("sigma", [math.nan, 0.0])
def test_no_z_score_without_a_positive_sigma(sigma: float) -> None:
    outcome = measure(camera(80.0, 65.0, sigma, IntrinsicsSource.ASSUMED_FOV))
    assert outcome.samples["fov_z"].size == 0
    assert outcome.samples["fov_err_deg"][0] == pytest.approx(5.0)


def test_compute_summarizes_without_context() -> None:
    summary = FovMetric(OPTIONS).compute(
        camera(80.0, 62.0, math.nan, IntrinsicsSource.ASSUMED_FOV), truth(), manifest()
    )
    assert summary["fov_err_deg.mean"] == pytest.approx(2.0)
    assert summary["fov_z.n"] == 0.0
