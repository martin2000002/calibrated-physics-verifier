"""Relative sigma of the characteristic length L = (e_x e_y e_z)^(1/3) from the relative sigmas
r_j = sigma_e_j / e_j of its three extents: d ln L = (1/3) sum_j d ln e_j.

A perceived lift scales every extent of an object by one depth and one focal factor, and its
e_z is the mean of the lateral extents (``backprojection.full_extents``), so the axis errors are
correlated: sigma_L / L = (1/3) sum_j r_j, the mean of the r_j ("correlated"). Only the own-size
error q is drawn per axis (``backprojection.monte_carlo_sigma``); with a common term c and q
independent per axis the exact value is sqrt(c^2 + q^2 / 3), and the mean, sqrt(c^2 + q^2), is
an upper bound. The split of r_j into c and q is not in the extent sigmas, so the bound is used:
with the lift's error profile (c ~ 0.40, q ~ 0.08) it is about 1.3 % high [estimate].

"independent" is the earlier formula (1/3) sqrt(sum_j r_j^2), kept to reproduce earlier runs; it
is short by sqrt(3) when the three r_j are one shared error."""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

AxisErrors = Literal["correlated", "independent"]
DEFAULT_AXIS_ERRORS: Final[AxisErrors] = "correlated"


def length_rel_sigma(
    rel_extent: NDArray[np.float64], axis_errors: AxisErrors
) -> NDArray[np.float64]:
    """sigma_L / L over the last axis (3) of ``rel_extent``; NaN propagates."""
    if axis_errors == "correlated":
        mean: NDArray[np.float64] = np.mean(rel_extent, axis=-1)
        return mean
    quadrature: NDArray[np.float64] = np.sqrt(np.sum(rel_extent**2, axis=-1)) / 3.0
    return quadrature
