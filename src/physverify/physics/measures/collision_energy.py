"""R3 collisions: restitution [e_minus_1], mass ratio from momentum, energy gain [dE_over_E]."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from physverify.physics.geometry import as_vector


def restitution(v_before: ArrayLike, v_after: ArrayLike, normal: ArrayLike) -> float:
    """e = -(v+ . n) / (v- . n) with relative velocities along the contact normal."""
    n = as_vector(normal)
    return -float(as_vector(v_after) @ n) / float(as_vector(v_before) @ n)


def momentum_mass_ratio(dv_i: ArrayLike, dv_j: ArrayLike) -> tuple[float, float]:
    """Mass ratio m_i / m_j = |dv_j| / |dv_i| implied by momentum conservation, and the cosine
    of the angle between dv_i and -dv_j (1 when the impulses are exactly opposite)."""
    a, b = as_vector(dv_i), as_vector(dv_j)
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return norm_b / norm_a, float(a @ -b) / (norm_a * norm_b)


def kinetic_energy_gain(
    mass_ratio: float,
    v_before_i: ArrayLike,
    v_before_j: ArrayLike,
    v_after_i: ArrayLike,
    v_after_j: ArrayLike,
) -> float:
    """(E+ - E-) / E- with E = r |v_i|^2 + |v_j|^2 (masses in units of m_j, r = m_i / m_j);
    NaN when nothing moves before the collision (E- = 0)."""
    before = incoming_energy(mass_ratio, v_before_i, v_before_j)
    if before == 0.0:
        return math.nan
    after = mass_ratio * _sq(v_after_i) + _sq(v_after_j)
    return (after - before) / before


def incoming_energy(mass_ratio: float, v_before_i: ArrayLike, v_before_j: ArrayLike) -> float:
    """E- = r |v_i|^2 + |v_j|^2, the kinetic energy before the collision in units of m_j / 2."""
    return mass_ratio * _sq(v_before_i) + _sq(v_before_j)


def _sq(values: ArrayLike) -> float:
    vector = as_vector(values)
    return float(vector @ vector)
