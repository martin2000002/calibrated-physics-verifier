"""Scale-free normalizers for Level 1 residuals.

The characteristic length and g0 are defined once in ``physics.geometry`` and
``schema.constants``; they are re-exported here, never redefined.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from physverify.physics.geometry import characteristic_length, characteristic_lengths
from physverify.schema.constants import STANDARD_GRAVITY

__all__ = [
    "STANDARD_GRAVITY",
    "characteristic_length",
    "characteristic_lengths",
    "gravity_in_units",
    "reference_speed",
]


def reference_speed(velocity: NDArray[np.float64], length: float, duration: float) -> float:
    """v_ref = max(RMS_t ||v_gt||, L_gt / T) over the finite rows of ``velocity`` (..., 3)."""
    speeds = np.linalg.norm(np.asarray(velocity, dtype=np.float64), axis=-1).ravel()
    finite = speeds[np.isfinite(speeds)]
    rms = float(np.sqrt(np.mean(finite**2))) if finite.size else math.nan
    floor = length / duration if duration > 0 and math.isfinite(length) else math.nan
    candidates = [value for value in (rms, floor) if math.isfinite(value)]
    return max(candidates) if candidates else math.nan


def gravity_in_units(meters_per_unit: float) -> float:
    """g0 / s in table units per s^2; NaN when the scale is not a finite positive number."""
    if not (math.isfinite(meters_per_unit) and meters_per_unit > 0):
        return math.nan
    return STANDARD_GRAVITY / meters_per_unit
