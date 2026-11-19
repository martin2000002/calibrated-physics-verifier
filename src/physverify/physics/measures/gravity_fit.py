"""R5 gravity: parabolic fall fit and shape [R5a], gravity value [R5b, g_rel_error]."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.schema.constants import STANDARD_GRAVITY

Vec = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class FallFit:
    """p(t) = p0 + v0 (t - t0) + a (t - t0)^2 / 2 fitted by least squares."""

    a: Vec
    v0: Vec
    p0: Vec
    t0: float
    rms: float
    n: int


def fall_fit(t: ArrayLike, positions: ArrayLike) -> FallFit:
    """Least-squares parabola over the frames with finite positions (at least three)."""
    times = np.asarray(t, dtype=np.float64).reshape(-1)
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    keep = np.isfinite(times) & np.all(np.isfinite(p), axis=1)
    times, p = times[keep], p[keep]
    if times.shape[0] < 3:
        raise ValueError("a fall fit needs at least three frames with known position")
    tau = times - times[0]
    design = np.column_stack([np.ones_like(tau), tau, 0.5 * tau**2])
    coef, *_ = np.linalg.lstsq(design, p, rcond=None)
    residual = p - design @ coef
    rms = math.sqrt(float(np.mean(np.sum(residual**2, axis=1))))
    return FallFit(a=coef[2], v0=coef[1], p0=coef[0], t0=float(times[0]), rms=rms, n=len(tau))


def fall_residual_ratio(fit: FallFit, duration: float) -> float:
    """rms / (|a| T^2 / 2): fit residual over the distance fallen in the window."""
    return fit.rms / (0.5 * float(np.linalg.norm(fit.a)) * duration**2)


def accel_inconsistency(accelerations: Sequence[ArrayLike]) -> float:
    """max_ij |a_i - a_j| / median |a| over the fitted accelerations of several objects or
    windows [accel_rel_diff]."""
    vectors = [np.asarray(a, dtype=np.float64).reshape(3) for a in accelerations]
    if len(vectors) < 2:
        raise ValueError("need at least two accelerations")
    spread = max(float(np.linalg.norm(a - b)) for a, b in itertools.combinations(vectors, 2))
    return spread / float(np.median([np.linalg.norm(a) for a in vectors]))


def lateral_accel_ratio(acceleration: ArrayLike, gravity_dir: ArrayLike) -> float:
    """|a - (a . g) g| / |a|: share of the acceleration orthogonal to the gravity direction."""
    a = np.asarray(acceleration, dtype=np.float64).reshape(3)
    g = np.asarray(gravity_dir, dtype=np.float64).reshape(3)
    g = g / np.linalg.norm(g)
    lateral = a - float(a @ g) * g
    return float(np.linalg.norm(lateral)) / float(np.linalg.norm(a))


def gravity_rel_error(accel_norm: float, meters_per_unit: float) -> float:
    """| |a| s - g0 | / g0 with |a| in table units per s^2 and s in meters per unit."""
    return abs(accel_norm * meters_per_unit - STANDARD_GRAVITY) / STANDARD_GRAVITY
