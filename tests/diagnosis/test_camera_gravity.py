"""it2 T13: camera/gravity, the perceived pose's gravity against the true one."""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
import pytest
from numpy.typing import NDArray
from synthetic_clip import manifest

from physverify.cli.context import PLUGIN_KINDS
from physverify.diagnosis.cases import MetricCase, MetricOutcome
from physverify.diagnosis.error_profile_writer import PROFILE_QUANTITIES
from physverify.diagnosis.metrics.camera_gravity import CameraGravityMetric
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.providers._gravity_pose import pose_from_gravity
from physverify.schema.clip import FrameSampling
from physverify.schema.perception import CameraModel, IntrinsicsSource
from physverify.synth.layout import camera_axes

OPTIONS = MetricSettings(
    boundary_tol_frac=0.008,
    depth_alignment="clip",
    depth_eps_frac=1e-3,
    contact_min_tiou=0.3,
    impact_half_window_s=0.125,
)
DECLARED_POSE_SIGMA_DEG = 2.0
TOL = 1e-9
SAMPLING = FrameSampling(
    frame_indices=np.arange(3, dtype=np.int64),
    width=64,
    height=48,
    native_width=64,
    native_height=48,
    seconds_per_frame=1 / 24,
    step=1,
    max_side=None,
)


def tilted_pose(tilt_deg: float) -> NDArray[np.float64]:
    tilt = math.radians(tilt_deg)
    pose = np.eye(4)
    pose[:3, :3] = camera_axes(
        np.array([0.0, -4 * math.cos(tilt), 4 * math.sin(tilt)]), np.zeros(3)
    )
    return pose


def camera(pose: NDArray[np.float64] | None, source: IntrinsicsSource) -> CameraModel:
    k = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]])
    return CameraModel(SAMPLING, k, source, 56.0, 0.0, world_from_camera=pose)


def measure(pred: CameraModel, truth_tilt_deg: float) -> MetricOutcome:
    case = MetricCase(
        clip=manifest(),
        pred=pred,
        truth=camera(tilted_pose(truth_tilt_deg), IntrinsicsSource.GROUND_TRUTH),
        context=MappingProxyType({}),
        gt=None,
        options=OPTIONS,
        plan=None,
    )
    return CameraGravityMetric(OPTIONS).samples(case)


def test_t13_a_true_pose_has_zero_angle() -> None:
    outcome = measure(camera(tilted_pose(25.0), IntrinsicsSource.ESTIMATED), 25.0)
    assert outcome.samples["gravity_angle_deg"] == pytest.approx([0.0], abs=1e-6)
    assert outcome.samples["gravity_z"] == pytest.approx([0.0], abs=1e-6)


def test_t13_without_a_pose_the_angle_is_empty_and_the_level_angle_is_the_tilt() -> None:
    outcome = measure(camera(None, IntrinsicsSource.ASSUMED_FOV), 25.0)
    assert outcome.samples["gravity_angle_deg"].size == 0
    assert outcome.samples["gravity_z"].size == 0
    assert outcome.samples["level_angle_deg"] == pytest.approx([25.0], abs=TOL)


def test_t13_an_estimated_pose_off_by_four_degrees_is_two_declared_sigma() -> None:
    tilt = math.radians(29.0)
    pose = pose_from_gravity(np.array([0.0, math.cos(tilt), math.sin(tilt)]))
    outcome = measure(camera(pose, IntrinsicsSource.ESTIMATED), 25.0)
    assert outcome.samples["gravity_angle_deg"] == pytest.approx([4.0], abs=TOL)
    assert outcome.samples["gravity_z"] == pytest.approx([4.0 / DECLARED_POSE_SIGMA_DEG], abs=TOL)


def test_the_metric_is_registered_and_stays_out_of_the_profile() -> None:
    assert PLUGIN_KINDS["metrics"]()["camera/gravity"] is CameraGravityMetric
    assert "camera/gravity" not in PROFILE_QUANTITIES
