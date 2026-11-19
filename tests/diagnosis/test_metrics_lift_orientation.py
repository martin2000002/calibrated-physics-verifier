"""lift/orientation: the pose error counted up to the true solid's symmetry."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from metric_cases import METRIC_OPTIONS as OPTIONS
from metric_cases import metric_case
from numpy.typing import NDArray

from physverify.diagnosis.error_profile_writer import PROFILE_QUANTITIES
from physverify.diagnosis.measurement_cases import MetricOutcome
from physverify.diagnosis.metrics.lift_orientation import LiftOrientationMetric
from physverify.physics.geometry import quaternion_from_matrix
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable

TOL = 1e-6
FloatArray = NDArray[np.float64]


def about(axis: int, degrees: float) -> FloatArray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    i, j = [k for k in range(3) if k != axis]
    rotation = np.eye(3)
    rotation[i, i], rotation[i, j], rotation[j, i], rotation[j, j] = c, -s, s, c
    return rotation


def table(primitive: str, size: FloatArray, rotation: FloatArray, sigma: float) -> ObjectStateTable:
    blank = ObjectStateTable.blank(
        clip_id="clip",
        frame=CoordFrame.CAMERA_OPENCV,
        unit=LengthUnit.METER,
        source_key="lift:clip",
        t=np.arange(2) / 24.0,
        frame_index=np.arange(2, dtype=np.int64),
        object_ids=["obj"],
        labels=[primitive],
        primitive=[primitive],
    )
    return dataclasses.replace(
        blank,
        quaternion=np.tile(quaternion_from_matrix(rotation), (2, 1, 1)),
        orientation_sigma=np.full((2, 1), sigma),
        size=np.tile(size, (2, 1, 1)),
        visible=np.ones((2, 1), dtype=bool),
    )


def measure(primitive: str, size: FloatArray, pred: FloatArray, truth: FloatArray) -> MetricOutcome:
    case = metric_case(table(primitive, size, pred, 0.1), table(primitive, size, truth, 0.0))
    return LiftOrientationMetric(OPTIONS).samples(case)


def test_a_cube_turned_a_quarter_about_any_axis_is_the_same_cube() -> None:
    base = about(2, 17.0)
    for axis in range(3):
        outcome = measure("box", np.full(3, 0.4), base @ about(axis, 90.0), base)
        assert outcome.samples["orientation_rad"] == pytest.approx([0.0, 0.0], abs=TOL)


def test_a_long_box_turned_a_quarter_is_off_by_a_quarter() -> None:
    size = np.array([0.5, 0.4, 0.3])
    outcome = measure("box", size, about(2, 90.0), np.eye(3))
    assert outcome.samples["orientation_rad"] == pytest.approx([math.pi / 2] * 2, abs=TOL)
    turned = measure("box", size, about(2, 180.0) @ about(0, 10.0), np.eye(3))
    assert turned.samples["orientation_rad"] == pytest.approx([math.radians(10.0)] * 2, abs=TOL)
    assert turned.samples["orientation_z"] == pytest.approx([math.radians(10.0) / 0.1] * 2)


def test_a_cylinder_counts_only_its_axis_and_a_sphere_nothing() -> None:
    size = np.array([0.3, 0.3, 0.5])
    spun = measure("cylinder", size, about(2, 63.0) @ about(0, 180.0), np.eye(3))
    assert spun.samples["orientation_rad"] == pytest.approx([0.0, 0.0], abs=TOL)
    tilted = measure("cylinder", size, about(1, 12.0), np.eye(3))
    assert tilted.samples["orientation_rad"] == pytest.approx([math.radians(12.0)] * 2)
    ball = measure("sphere", np.full(3, 0.2), about(1, 40.0), np.eye(3))
    assert ball.samples["orientation_rad"].size == 0


def test_an_unknown_pose_gives_no_sample_and_the_angle_enters_the_profile() -> None:
    pred = table("box", np.full(3, 0.4), np.eye(3), math.nan)
    pred = dataclasses.replace(pred, quaternion=np.full((2, 1, 4), np.nan))
    truth = table("box", np.full(3, 0.4), np.eye(3), 0.0)
    outcome = LiftOrientationMetric(OPTIONS).samples(metric_case(pred, truth))
    assert outcome.samples["orientation_rad"].size == 0
    assert PROFILE_QUANTITIES["lift/orientation"] == ("orientation_rad",)
