"""R1 interpenetration magnitude [pen_over_size]."""

from __future__ import annotations

import math

from numpy.typing import ArrayLike

from physverify.physics.geometry import characteristic_length


def penetration_ratio(distance: float, extents_a: ArrayLike, extents_b: ArrayLike) -> float:
    """max(0, -d) / min(L_a, L_b): penetration depth over the smaller characteristic length."""
    if math.isnan(distance):
        return math.nan
    smaller = min(characteristic_length(extents_a), characteristic_length(extents_b))
    return max(0.0, -distance) / smaller
