"""Which disturbance of the positions the whitened third differences around a peak look like: a
step (the body is displaced and stays there, a teleport), a kink (its velocity changes at one
frame, an impact) or an outlier (one frame is off, a detector or timing glitch).

Row q of ``whitened_jumps`` is the stencil -1, 3, -3, 1 on points q-1..q+2 over sqrt(20) sigma, so
a disturbance of profile p gives the response sum_i c_i p(q-1+i) / sqrt(20) on row q: [1, -2, 1]
for a step, [1, -1] for a kink, [1, -3, 3, -1] for an outlier (up to sign and scale). Overlapping
stencils share points, so the noise of adjacent rows is correlated (``row_correlation``) and each
shape is fitted by generalized least squares, one free amplitude per axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.physics.ties import TIE_RTOL

FloatArray = NDArray[np.float64]
Shape = Literal["step", "kink", "outlier"]
SHAPES: tuple[Shape, ...] = ("outlier", "kink", "step")
"""In the order ties resolve to: a window that cannot tell a step from the others (a peak whose
neighbouring stencils are excluded fits every shape exactly) is not read as a step."""
STENCIL = np.array([-1.0, 3.0, -3.0, 1.0])
STEP_ROWS = 3
"""Rows a step's response spans ([1, -2, 1]): a window with fewer finite rows cannot show one."""
REACH = 3
"""Rows on each side of the peak in the fit: every placement whose response touches the peak
row lies within it."""


@dataclass(frozen=True)
class ShapeFit:
    """Reduction of the GLS chi-square of the window by the best placement of each shape, and
    how many rows the window had."""

    gains: dict[Shape, float]
    rows: int

    @property
    def best(self) -> Shape:
        """The shape of largest gain; within ``TIE_RTOL`` of it, the earliest in ``SHAPES``; with
        fewer than ``STEP_ROWS`` rows, the best other than a step."""
        if self.rows < STEP_ROWS:
            others = [shape for shape in SHAPES if shape != "step"]
            return max(others, key=lambda shape: self.gains[shape])
        top = max(self.gains.values())
        return next(s for s in SHAPES if self.gains[s] >= top * (1.0 - TIE_RTOL))


def row_correlation(lag: int) -> float:
    """Correlation of the whitened third differences of rows ``lag`` apart under white noise."""
    lag = abs(lag)
    if lag >= STENCIL.size:
        return 0.0
    return float(STENCIL[: STENCIL.size - lag] @ STENCIL[lag:] / (STENCIL @ STENCIL))


def profile(shape: Shape, at: int, points: FloatArray) -> FloatArray:
    """Unit disturbance of the positions at ``points``, placed at point ``at``."""
    if shape == "step":
        return np.asarray(points > at, dtype=np.float64)
    if shape == "kink":
        return np.asarray(np.maximum(points - at, 0.0), dtype=np.float64)
    return np.asarray(points == at, dtype=np.float64)


def response(shape: Shape, at: int, rows: FloatArray) -> FloatArray:
    """Whitened third differences, at ``rows``, of the unit disturbance ``shape`` at ``at``."""
    points = rows[:, None] - 1.0 + np.arange(STENCIL.size)[None, :]
    return np.asarray(profile(shape, at, points) @ STENCIL / np.sqrt(STENCIL @ STENCIL))


def fit_shapes(whitened: ArrayLike, row: int) -> ShapeFit:
    """Best GLS gain of each shape over the finite rows within ``REACH`` of ``row``, over the
    placements whose response is not zero on ``row``."""
    values = np.asarray(whitened, dtype=np.float64).reshape(-1, 3)
    near = np.arange(max(row - REACH, 0), min(row + REACH + 1, values.shape[0]))
    rows = near[np.all(np.isfinite(values[near]), axis=1)].astype(np.float64)
    window = values[rows.astype(int)]
    lags = rows[:, None] - rows[None, :]
    precision = np.linalg.inv(np.vectorize(row_correlation)(lags.astype(int)))
    gains: dict[Shape, float] = {}
    for shape in SHAPES:
        gains[shape] = max(
            (
                _gain(window, precision, r)
                for at in range(row - REACH, row + REACH + 1)
                if (r := response(shape, at, rows))[rows == row].any()
            ),
            default=0.0,
        )
    return ShapeFit(gains, int(rows.size))


def _gain(window: FloatArray, precision: FloatArray, r: FloatArray) -> float:
    """chi-square reduction of fitting amplitude * ``r`` on every axis of ``window``."""
    weight = float(r @ precision @ r)
    if weight <= 0.0:
        return 0.0
    projection = window.T @ precision @ r
    return float(projection @ projection / weight)
