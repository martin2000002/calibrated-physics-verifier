"""R4_jump v2 [jump_over_noise]: the third difference of one object's track in units of its own
position noise (plan-it4 change A). Inputs are the time series of a single object (time on axis
0) and its per-axis noise sigma-hat."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.physics.finite_diff import third_difference, third_difference_sigma

FloatArray = NDArray[np.float64]


def noise_jumps(
    positions: ArrayLike, sigma_hat: ArrayLike, exclude: ArrayLike | None = None
) -> FloatArray:
    """Per row k: ||D3_k / (sqrt(20) sigma_hat)||_2 of the stencil rows k-1..k+2, NaN where the
    stencil does not exist (first row, last two), has an unknown position or is ``exclude``-d.

    The third difference vanishes under any constant acceleration and the ratio under any common
    rescaling of positions and sigma_hat, so the value needs neither g nor a scale. Under white
    noise of sigma_hat each axis term is a standard normal; a step d between rows j and j + 1 gives
    2 d / (sqrt(20) sigma_hat) at row j."""
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    out = np.full(p.shape[0], math.nan)
    if p.shape[0] < 4:
        return out
    per_unit = third_difference_sigma(1.0, 1.0)  # sqrt(20): sigma of D3 per unit position sigma
    scale = per_unit * np.asarray(sigma_hat, dtype=np.float64).reshape(3)
    values = np.linalg.norm(third_difference(p) / scale, axis=1)
    if exclude is not None:
        values[np.asarray(exclude, dtype=bool).reshape(-1)[1:-2]] = math.nan
    out[1:-2] = values
    return out


def jump_over_noise(
    positions: ArrayLike, sigma_hat: ArrayLike, exclude: ArrayLike | None = None
) -> float:
    """max_k of ``noise_jumps``; NaN when no stencil qualifies."""
    values = noise_jumps(positions, sigma_hat, exclude)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else math.nan
