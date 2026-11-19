"""The window covariance shared by the line-of-sight variants of R5a and R5b."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from physverify.checks.variants._whitened import exact_floor
from physverify.physics.flights import Flight
from physverify.physics.line_of_sight import line_of_sight_covariance
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]


def window_covariance(
    states: ObjectStateTable, window: Flight, sigma: FloatArray, lengths: float
) -> FloatArray:
    """(n, 3, 3) Sigma_t of the window rows from their floored ``sigma``: the rows' own sigma
    tells which entries are exact, and the floor is ``exact_floor`` of ``lengths``."""
    raw = np.asarray(states.position_sigma[window.rows, window.index], dtype=np.float64)
    position = states.position[window.rows, window.index]
    floor = exact_floor(states, window.index, lengths)
    return line_of_sight_covariance(position, sigma, raw == 0.0, states.frame, floor)


def known_gravity(states: ObjectStateTable) -> bool:
    return bool(np.all(np.isfinite(states.gravity_dir)))


def known_gravity_sigma(states: ObjectStateTable) -> bool:
    return known_gravity(states) and math.isfinite(states.gravity_dir_sigma)
