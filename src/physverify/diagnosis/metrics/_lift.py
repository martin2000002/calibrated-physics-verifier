"""The LIFT back-projection params a metric reads, and gravity angles in camera axes."""

from __future__ import annotations

import numpy as np

from physverify.diagnosis.measurement_cases import FloatArray
from physverify.diagnosis.metrics._geometry import angle_deg
from physverify.extraction.lift_params import LiftParams
from physverify.extraction.plan import ResolvedPlan
from physverify.schema.stages import Stage


def lift_params(plan: ResolvedPlan | None) -> LiftParams:
    """The plan's back-projection params, so the depth is read exactly as LIFT reads it."""
    if plan is not None and Stage.LIFT in plan.names:
        params = plan.step(Stage.LIFT).params
        if isinstance(params, LiftParams):
            return params
    return LiftParams()


def gravity_angles(assumed: FloatArray, truth: FloatArray) -> FloatArray:
    """[angle in degrees], or empty when either direction is unknown (non-finite)."""
    if not (np.isfinite(assumed).all() and np.isfinite(truth).all()):
        return np.zeros(0)
    return np.array([angle_deg(assumed, truth)])
