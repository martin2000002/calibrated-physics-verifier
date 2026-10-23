"""LIFT: angle between the gravity direction LIFT assumes and the true one, in camera axes."""

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
from physverify.diagnosis.metrics.contacts_events import angle_deg
from physverify.diagnosis.metrics.depth_object import lift_params
from physverify.diagnosis.settings import MetricSettings
from physverify.extraction.providers._gravity_pose import LEVEL_CAMERA_GRAVITY
from physverify.schema.clip import ClipManifest
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable


def gravity_angles(assumed: FloatArray, truth: FloatArray) -> FloatArray:
    """[angle in degrees], or empty when either direction is unknown (non-finite)."""
    if not (np.isfinite(assumed).all() and np.isfinite(truth).all()):
        return np.zeros(0)
    return np.array([angle_deg(assumed, truth)])


@dataclass(frozen=True)
class LiftGravityAngleMetric:
    """One sample per clip. ``gravity_angle_deg``: the measured LIFT's ``gravity_dir`` against
    the truth (0 when it read an exact pose). ``level_angle_deg``: the level-camera +y that the
    perceived chain assumes without a pose; ``level_z`` divides it by the declared
    ``level_sigma_deg``. Diagnostic only, so it is not in ``PROFILE_QUANTITIES``."""

    stage: ClassVar[Stage] = Stage.LIFT
    name: ClassVar[str] = "gravity_angle"
    options: MetricSettings

    def compute(
        self, pred: StageOutput, truth: StageOutput, clip: ClipManifest
    ) -> Mapping[str, float]:
        return bare_compute(self, pred, truth, clip, self.options)

    def samples(self, case: MetricCase) -> MetricOutcome:
        pred, truth = case.pred, case.truth
        if not isinstance(pred, ObjectStateTable) or not isinstance(truth, ObjectStateTable):
            raise TypeError("lift metric compares two ObjectStateTable")
        level = gravity_angles(LEVEL_CAMERA_GRAVITY, truth.gravity_dir)
        sigma_deg = lift_params(case.plan).level_sigma_deg
        level_z = level / sigma_deg if sigma_deg > 0 else np.zeros(0)
        samples = {
            "gravity_angle_deg": gravity_angles(pred.gravity_dir, truth.gravity_dir),
            "level_angle_deg": level,
            "level_z": level_z,
        }
        return MetricOutcome(samples=MappingProxyType(samples))


PLUGINS = (LiftGravityAngleMetric,)
