"""Vector geometry the metrics share."""

from __future__ import annotations

import math

import numpy as np

from physverify.diagnosis.measurement_cases import FloatArray


def angle_deg(u: FloatArray, v: FloatArray) -> float:
    """Angle via atan2(|u x v|, u . v): exact near 0, where arccos loses half the digits."""
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(np.dot(u, v))))
