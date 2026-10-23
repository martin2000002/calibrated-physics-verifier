"""Closed-form sigma of the three-point differences, the oracle of the propagated sigma."""

from __future__ import annotations

import math


def central_difference_sigma(sigma_x: float, dt: float) -> float:
    """Sigma of (x+ - x-) / (2 dt) for independent position noise sigma_x."""
    return sigma_x / (math.sqrt(2.0) * dt)


def second_difference_sigma(sigma_x: float, dt: float) -> float:
    """Sigma of (x+ - 2x + x-) / dt^2 for independent position noise: sqrt(6) sigma_x / dt^2."""
    return math.sqrt(6.0) * sigma_x / dt**2
