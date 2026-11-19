"""Scale-free normalizers for Level 1 residuals.

The characteristic length and g0 are defined once in ``physics.geometry`` and
``schema.constants``; they are re-exported here, never redefined.
"""

from __future__ import annotations

import math

from physverify.physics.geometry import characteristic_length, characteristic_lengths
from physverify.physics.kinematics import reference_speed
from physverify.schema.constants import STANDARD_GRAVITY

__all__ = [
    "STANDARD_GRAVITY",
    "characteristic_length",
    "characteristic_lengths",
    "gravity_in_units",
    "reference_speed",
]


def gravity_in_units(meters_per_unit: float) -> float:
    """g0 / s in table units per s^2; NaN when the scale is not a finite positive number."""
    if not (math.isfinite(meters_per_unit) and meters_per_unit > 0):
        return math.nan
    return STANDARD_GRAVITY / meters_per_unit
