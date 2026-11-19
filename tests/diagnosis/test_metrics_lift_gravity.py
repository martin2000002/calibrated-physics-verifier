"""lift/gravity_angle: the level-camera gravity against the true one, on a camera of known pitch."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from metric_cases import METRIC_OPTIONS as OPTIONS
from metric_cases import metric_case
from numpy.typing import NDArray
from synthetic_clip import manifest

from physverify.app.plugins import PLUGIN_KINDS
from physverify.diagnosis.error_profile_writer import PROFILE_QUANTITIES
from physverify.diagnosis.measurement_cases import MetricOutcome
from physverify.diagnosis.metrics.lift_gravity import LiftGravityAngleMetric
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable
from physverify.synth.placement import camera_axes

KEY = "lift/gravity_angle"
WORLD_DOWN = np.array([0.0, 0.0, -1.0])
DECLARED_LEVEL_SIGMA_DEG = 10.0
TOL = 1e-9


def camera_gravity(pitch_deg: float, azimuth_deg: float = 30.0) -> NDArray[np.float64]:
    """World down in the OpenCV axes of a camera at ``pitch_deg`` elevation looking at 0."""
    pitch, azimuth = math.radians(pitch_deg), math.radians(azimuth_deg)
    position = 4.0 * np.array(
        [math.cos(pitch) * math.cos(azimuth), math.cos(pitch) * math.sin(azimuth), math.sin(pitch)]
    )
    rotation = camera_axes(position, np.zeros(3))
    gravity: NDArray[np.float64] = rotation.T @ WORLD_DOWN
    return gravity


def table(gravity_dir: NDArray[np.float64], unit: LengthUnit) -> ObjectStateTable:
    blank = ObjectStateTable.blank(
        clip_id="clip",
        frame=CoordFrame.CAMERA_OPENCV,
        unit=unit,
        source_key="lift:clip",
        t=np.arange(3) / 24.0,
        frame_index=np.arange(3, dtype=np.int64),
        object_ids=["ball"],
        labels=["ball"],
        primitive=["sphere"],
    )
    return dataclasses.replace(blank, gravity_dir=gravity_dir, gravity_dir_sigma=0.0)


def measure(pred_gravity: NDArray[np.float64], pitch_deg: float) -> MetricOutcome:
    case = metric_case(
        table(pred_gravity, LengthUnit.RELATIVE), table(camera_gravity(pitch_deg), LengthUnit.METER)
    )
    return LiftGravityAngleMetric(OPTIONS).samples(case)


@pytest.mark.parametrize("pitch_deg", [15.0, 25.0, 35.0])
def test_level_camera_angle_equals_the_camera_pitch(pitch_deg: float) -> None:
    outcome = measure(camera_gravity(pitch_deg), pitch_deg)
    assert outcome.samples["level_angle_deg"] == pytest.approx([pitch_deg], abs=TOL)
    assert outcome.samples["gravity_angle_deg"] == pytest.approx([0.0], abs=TOL)


def test_level_z_divides_by_the_declared_level_sigma() -> None:
    outcome = measure(camera_gravity(25.0), 25.0)
    assert outcome.samples["level_z"] == pytest.approx([25.0 / DECLARED_LEVEL_SIGMA_DEG])


def test_a_level_camera_prediction_measures_the_pitch() -> None:
    outcome = measure(np.array([0.0, 1.0, 0.0]), 25.0)
    assert outcome.samples["gravity_angle_deg"] == pytest.approx([25.0], abs=TOL)


def test_unknown_predicted_gravity_gives_no_sample() -> None:
    outcome = measure(np.full(3, np.nan), 25.0)
    assert outcome.samples["gravity_angle_deg"].size == 0
    assert outcome.samples["level_angle_deg"] == pytest.approx([25.0], abs=TOL)


def test_compute_summarizes_one_sample_per_clip() -> None:
    pred = table(np.array([0.0, 1.0, 0.0]), LengthUnit.RELATIVE)
    truth = table(camera_gravity(20.0), LengthUnit.METER)
    summary = LiftGravityAngleMetric(OPTIONS).compute(pred, truth, manifest())
    assert summary["level_angle_deg.mean"] == pytest.approx(20.0, abs=TOL)
    assert summary["level_angle_deg.n"] == 1.0


def test_registered_and_kept_out_of_the_error_profile() -> None:
    assert PLUGIN_KINDS["metrics"]()[KEY] is LiftGravityAngleMetric
    assert KEY not in PROFILE_QUANTITIES
