"""Savitzky-Golay derivatives of sampled trajectories with closed-form sigma (ADR 0005, 4.8).

Each derivative is a linear operator W applied to the positions of one visible run, so with
independent position noise sigma_x the derivative sigma is sqrt(W^2 sigma_x^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_coeffs

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

UNIFORM_STEP_RTOL = 1e-6


@dataclass(frozen=True)
class SavgolSpec:
    window: int
    polyorder: int

    def __post_init__(self) -> None:
        if self.window % 2 == 0 or self.window <= self.polyorder or self.polyorder < 0:
            raise ValueError("window must be odd and > polyorder >= 0")


@dataclass(frozen=True)
class TrackDerivatives:
    velocity: FloatArray
    acceleration: FloatArray
    velocity_sigma: FloatArray
    acceleration_sigma: FloatArray


def fitted_window(n: int, spec: SavgolSpec) -> int | None:
    """Largest odd window <= min(n, spec.window) that still exceeds the polynomial order."""
    window = min(n, spec.window)
    if window % 2 == 0:
        window -= 1
    return window if window > spec.polyorder else None


def savgol_operator(n: int, spec: SavgolSpec, deriv: int, dt: float) -> FloatArray:
    """(n, n) matrix W with W @ x = d^deriv x / dt^deriv; NaN rows when n is too short."""
    window = fitted_window(n, spec)
    if window is None:
        return np.full((n, n), np.nan)
    half = window // 2
    operator = np.zeros((n, n))
    for row in range(n):
        first = min(max(row - half, 0), n - window)
        coeffs = savgol_coeffs(
            window, spec.polyorder, deriv=deriv, delta=dt, pos=row - first, use="dot"
        )
        operator[row, first : first + window] = coeffs
    return operator


def propagate_linear_sigma(operator: FloatArray, sigma: FloatArray) -> FloatArray:
    """sqrt(W^2 @ sigma^2); a NaN sigma poisons only the rows that give it weight."""
    weights = operator**2
    variance = np.nan_to_num(sigma**2, nan=0.0)
    out: FloatArray = np.sqrt(weights @ variance)
    unknown = (weights > 0).astype(np.float64) @ np.isnan(sigma).astype(np.float64)
    out[unknown > 0] = np.nan
    out[np.isnan(operator).any(axis=1)] = np.nan
    return out


def visible_runs(valid: BoolArray) -> tuple[slice, ...]:
    """Maximal runs of consecutive True samples."""
    flags = np.concatenate([[False], np.asarray(valid, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(flags.astype(np.int8)))
    return tuple(slice(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True))


def uniform_step(t: FloatArray) -> float:
    steps = np.diff(np.asarray(t, dtype=np.float64))
    if steps.size == 0:
        raise ValueError("a single sample has no time step")
    dt = float(steps.mean())
    if dt <= 0 or float(np.max(np.abs(steps - dt))) > UNIFORM_STEP_RTOL * dt:
        raise ValueError("Savitzky-Golay needs a uniform, increasing time grid")
    return dt


def differentiate(
    t: FloatArray,
    position: FloatArray,
    position_sigma: FloatArray,
    valid: BoolArray,
    spec: SavgolSpec,
) -> TrackDerivatives:
    """Velocity and acceleration (T, 3) run by run; NaN outside runs and in too-short runs."""
    out = [np.full(position.shape, np.nan) for _ in range(4)]
    for run in visible_runs(valid):
        n = run.stop - run.start
        if n < 2:
            continue
        dt = uniform_step(t[run])
        for index, deriv in ((0, 1), (1, 2)):
            operator = savgol_operator(n, spec, deriv, dt)
            out[index][run] = operator @ position[run]
            out[index + 2][run] = propagate_linear_sigma(operator, position_sigma[run])
    return TrackDerivatives(*out)
