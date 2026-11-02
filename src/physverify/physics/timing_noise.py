"""Timing noise of a track: rows sampled at an instant off by delta_r rows.

A row whose instant is off by delta_r moves every axis of every body at once, by e_{r,a} ~
delta_r v_a. With delta white of scale sigma_delta (in rows of the table), the third difference
of axis a has variance 20 (sigma_a^2 + sigma_delta^2 v_a^2) where the speed v is locally
constant, so d3_a / (sqrt(20) sigma_eff) is a standard normal per stencil and axis, as
d3_a / (sqrt(20) sigma_a) is under position noise alone.

sigma_delta is fitted on the cells (stencil, axis) in motion: v above sqrt(20) sigma-hat, the noise
of the cell's own third difference, with sigma-hat the track's own noise. A lower bound lets
position noise choose the cells: on a still track with white noise, v > sigma-hat keeps about half
of the cells, those whose steps (and so whose |d3|) came out large, and fits a sigma_delta near one
row where there is none. Its limit: sigma-hat is the robust scale of the same stencils, so on a
track that moves at a nearly uniform speed over the whole clip sigma-hat has already absorbed
sigma_delta v, the median |z| of the moving cells is the normal one, g(0) ~ 0 and sigma_delta = 0:
a check that uses it then equals its base. The fit acts only when most of the track is slow or
still (a fall followed by rest), where sigma-hat is the position noise and the moving cells are
louder than it. An estimate per track, not the timing error of the video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from physverify.physics.finite_diff import third_difference_sigma
from physverify.physics.uncertainty import MAD_TO_SIGMA

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class TimingFit:
    """sigma_delta in rows (0 when not fitted) and the moving cells it was fitted on."""

    sigma_frames: float
    cells: int


def stencil_speed(positions: ArrayLike) -> FloatArray:
    """(T - 3, 3): per stencil j (rows j..j+3) and axis, the median |step| of its three steps; NaN
    where a step is unknown. The median keeps one outlying step (a jump, a repeated row) out of v.
    """
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if p.shape[0] < 4:
        return np.empty((0, 3), dtype=np.float64)
    steps = np.abs(np.diff(p, axis=0))
    return np.asarray(np.median(np.stack([steps[:-2], steps[1:-1], steps[2:]]), axis=0))


@dataclass(frozen=True)
class MovingCells:
    """The cells sigma_delta is fitted on: |d3| / sqrt(20), sigma-hat^2 of their axis and v^2."""

    magnitude: FloatArray
    variance: FloatArray
    speed_squared: FloatArray

    @property
    def count(self) -> int:
        return int(self.magnitude.shape[0])

    def excess(self, s: float) -> float:
        """g(s) = median(|d3| / (sqrt(20) sqrt(sigma_a^2 + s^2 v^2))) / MAD_TO_SIGMA - 1.

        The median is not centred: centred, it is the robust scale sigma-hat was fitted with,
        which gives the trivial root 0. g does not increase with s and tends to -1."""
        scale = np.sqrt(self.variance + s**2 * self.speed_squared)
        return float(np.median(self.magnitude / scale)) / MAD_TO_SIGMA - 1.0


def moving_cells(
    d3: ArrayLike, sigma_hat: ArrayLike, speed: ArrayLike, usable: ArrayLike
) -> MovingCells:
    """The cells of a ``usable`` stencil with a finite d3 and a finite speed above sqrt(20) times
    a positive sigma-hat of their axis."""
    d = np.asarray(d3, dtype=np.float64).reshape(-1, 3)
    sigma = np.asarray(sigma_hat, dtype=np.float64).reshape(3)
    v = np.asarray(speed, dtype=np.float64).reshape(-1, 3)
    rows = np.asarray(usable, dtype=np.bool_).reshape(-1, 1)
    per_unit = third_difference_sigma(1.0, 1.0)
    with np.errstate(invalid="ignore"):
        moving = rows & np.isfinite(v) & np.isfinite(d) & (sigma > 0.0) & (v > per_unit * sigma)
    return MovingCells(
        magnitude=np.abs(d[moving]) / per_unit,
        variance=np.broadcast_to(sigma**2, d.shape)[moving],
        speed_squared=v[moving] ** 2,
    )


def timing_sigma(
    d3: ArrayLike, sigma_hat: ArrayLike, speed: ArrayLike, usable: ArrayLike, min_cells: int
) -> TimingFit:
    """sigma_delta, the root of ``MovingCells.excess`` over ``moving_cells``; 0 with fewer than
    ``min_cells`` of them or when g(0) <= 0. The bracket doubles from one row until g < 0."""
    cells = moving_cells(d3, sigma_hat, speed, usable)
    if cells.count < min_cells or cells.excess(0.0) <= 0.0:
        return TimingFit(0.0, cells.count)
    high = 1.0
    while cells.excess(high) >= 0.0:
        high *= 2.0
    return TimingFit(float(brentq(cells.excess, 0.0, high)), cells.count)


def stencil_sigma(sigma_hat: ArrayLike, speed: ArrayLike, sigma_frames: float) -> FloatArray:
    """(T - 3, 3): sqrt(sigma_a^2 + (sigma_delta v)^2), sigma-hat where v is not finite."""
    sigma = np.asarray(sigma_hat, dtype=np.float64).reshape(3)
    v = np.asarray(speed, dtype=np.float64).reshape(-1, 3)
    timed = np.sqrt(sigma**2 + (sigma_frames * v) ** 2)
    return np.where(np.isfinite(v), timed, np.broadcast_to(sigma, v.shape))
