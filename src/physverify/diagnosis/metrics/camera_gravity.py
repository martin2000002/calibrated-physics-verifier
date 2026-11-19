"""CAMERA: angle between the gravity of the perceived camera pose and the true one."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from physverify.diagnosis.measurement_cases import (
    FloatArray,
    MetricCase,
    MetricOutcome,
    bare_compute,
)
from physverify.diagnosis.metrics._lift import gravity_angles, lift_params
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.gravity_pose import LEVEL_CAMERA_GRAVITY, gravity_from_pose
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import CameraModel
from physverify.schema.stages import Stage


def pose_gravity(camera: CameraModel) -> FloatArray:
    """Gravity in camera axes from the pose; NaN when the camera has none."""
    if camera.world_from_camera is None:
        return np.full(3, np.nan)
    return gravity_from_pose(camera.world_from_camera)


@dataclass(frozen=True)
class CameraGravityMetric:
    """One sample per clip. ``gravity_angle_deg``: the perceived pose's gravity against the
    truth (empty without a perceived pose). ``level_angle_deg``: the level-camera +y against the
    truth. ``gravity_z`` divides the angle by the ``pose_sigma_deg`` LIFT declares for an
    estimated pose, so its coverage says whether that sigma holds. Diagnostic only, so it is not
    in ``PROFILE_QUANTITIES``."""

    stage: ClassVar[Stage] = Stage.CAMERA
    name: ClassVar[str] = "gravity"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, CameraModel) or not isinstance(truth, CameraModel):
            raise TypeError("camera gravity metric compares two CameraModel")
        true_gravity = pose_gravity(truth)
        angle = gravity_angles(pose_gravity(pred), true_gravity)
        sigma_deg = lift_params(case.plan).pose_sigma_deg
        samples = {
            "gravity_angle_deg": angle,
            "level_angle_deg": gravity_angles(LEVEL_CAMERA_GRAVITY, true_gravity),
            "gravity_z": angle / sigma_deg if sigma_deg > 0 else np.zeros(0),
        }
        return MetricOutcome(samples=MappingProxyType(samples))


PLUGINS = (CameraGravityMetric,)
