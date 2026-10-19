"""Exact event-driven flight of a sphere over a flat floor (no numerical integration).

In flight p(t) = p0 + v0 (t - t0) + g (t - t0)^2 / 2 with g piecewise constant. A floor hit is
solved from the quadratic at z = floor + r and gives v_z+ = -e v_z-, tangential velocity kept.
Below ``rest_speed`` the sphere stays on the floor and rolls with constant deceleration.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vec = NDArray[np.float64]
EPS_T = 1e-12
_MAX_EVENTS = 256


@dataclass(frozen=True)
class Segment:
    t0: float
    p0: Vec
    v0: Vec
    g: Vec

    def position(self, t: float) -> Vec:
        dt = t - self.t0
        out: Vec = self.p0 + self.v0 * dt + 0.5 * self.g * dt * dt
        return out

    def velocity(self, t: float) -> Vec:
        out: Vec = self.v0 + self.g * (t - self.t0)
        return out


@dataclass(frozen=True)
class GravitySchedule:
    """Gravity ``values[i]`` on [times[i-1], times[i]); ``values`` has one more entry."""

    times: tuple[float, ...]
    values: tuple[Vec, ...]

    @classmethod
    def constant(cls, g: ArrayLike) -> GravitySchedule:
        return cls((), (np.asarray(g, dtype=np.float64),))

    def at(self, t: float) -> Vec:
        index = sum(1 for switch in self.times if t >= switch - EPS_T)
        return self.values[index]

    def next_switch(self, t: float) -> float:
        return next((s for s in self.times if s > t + EPS_T), math.inf)


@dataclass(frozen=True)
class Bounce:
    t: float
    position: Vec
    v_before: Vec
    v_after: Vec


@dataclass
class Flight:
    positions: Vec
    velocities: Vec
    bounces: list[Bounce] = field(default_factory=list)


def impact_velocity(v_frame: ArrayLike, t_frame: float, t_c: float, g: ArrayLike) -> Vec:
    """Velocity at t_c extrapolated ballistically from a frame sample."""
    out: Vec = np.asarray(v_frame, dtype=np.float64) + np.asarray(g, dtype=np.float64) * (
        t_c - t_frame
    )
    return out


def time_to_height(z0: float, vz: float, gz: float, height: float) -> float:
    """First tau > 0 at which a falling z(tau) = z0 + vz tau + gz tau^2 / 2 reaches ``height``."""
    a, b, c = 0.5 * gz, vz, z0 - height
    if abs(a) < 1e-15:
        return -c / b if b < 0 and -c / b > EPS_T else math.inf
    disc = b * b - 4 * a * c
    if disc < 0:
        return math.inf
    root = math.sqrt(disc)
    candidates = sorted(((-b - root) / (2 * a), (-b + root) / (2 * a)))
    for tau in candidates:
        if tau > EPS_T and vz + gz * tau < 0:
            return tau
    return math.inf


def integrate_sphere(
    p0: ArrayLike,
    v0: ArrayLike,
    t0: float,
    radius: float,
    schedule: GravitySchedule,
    restitution: float | Sequence[float],
    t_grid: ArrayLike,
    *,
    rest_speed: float = 0.15,
    roll_decel: float = 0.5,
    floor_z: float = 0.0,
) -> Flight:
    """Positions and velocities on ``t_grid`` (all >= t0) plus every floor bounce.

    ``restitution`` is one value or one per bounce (the last one repeats)."""
    times = np.asarray(t_grid, dtype=np.float64).reshape(-1)
    out_p = np.full((times.size, 3), np.nan)
    out_v = np.full((times.size, 3), np.nan)
    coefficients = [restitution] if isinstance(restitution, float | int) else list(restitution)
    rest_z = floor_z + radius
    t, p, v = float(t0), np.asarray(p0, dtype=np.float64), np.asarray(v0, dtype=np.float64)
    bounces: list[Bounce] = []
    t_end = float(times.max()) if times.size else t
    for _ in range(_MAX_EVENTS):
        g = schedule.at(t)
        if p[2] <= rest_z + 1e-9 and abs(v[2]) < rest_speed and g[2] <= 0.0:
            start = np.array([p[0], p[1], rest_z])
            _roll(times, t, start, np.array([v[0], v[1], 0.0]), radius, roll_decel, out_p, out_v)
            break
        segment = Segment(t, p.copy(), v.copy(), g)
        tau_hit = time_to_height(p[2], v[2], g[2], rest_z)
        t_next = min(t + tau_hit, schedule.next_switch(t), t_end + 1.0)
        _fill(times, (t, t_next), segment.position, segment.velocity, out_p, out_v)
        if t_next > t_end:
            break
        p, v = segment.position(t_next), segment.velocity(t_next)
        t = t_next
        if abs(t - (segment.t0 + tau_hit)) < 1e-12:
            p[2] = rest_z
            e = float(coefficients[min(len(bounces), len(coefficients) - 1)])
            after = np.array([v[0], v[1], -e * v[2]])
            bounces.append(Bounce(t, p.copy(), v.copy(), after))
            v = after
            if abs(after[2]) < rest_speed:
                _roll(times, t, p, after, radius, roll_decel, out_p, out_v)
                break
    return Flight(out_p, out_v, bounces)


def _fill(
    times: Vec,
    window: tuple[float, float],
    position: Callable[[float], Vec],
    velocity: Callable[[float], Vec],
    out_p: Vec,
    out_v: Vec,
) -> None:
    lo, hi = window
    for k in np.flatnonzero((times >= lo - EPS_T) & (times < hi)):
        out_p[k] = position(float(times[k]))
        out_v[k] = velocity(float(times[k]))


def _roll(
    times: Vec, t0: float, p0: Vec, v0: Vec, radius: float, decel: float, out_p: Vec, out_v: Vec
) -> None:
    """Rolling on the floor: horizontal speed decreases at ``decel`` until the sphere stops."""
    horizontal = np.array([v0[0], v0[1], 0.0])
    speed = float(np.linalg.norm(horizontal))
    direction = horizontal / speed if speed > 0 else np.zeros(3)
    t_stop = speed / decel if decel > 0 else math.inf
    base = np.array([p0[0], p0[1], p0[2]])
    del radius
    for k in np.flatnonzero(times >= t0 - EPS_T):
        tau = min(float(times[k]) - t0, t_stop)
        out_p[k] = base + direction * (speed * tau - 0.5 * decel * tau * tau)
        out_v[k] = direction * max(0.0, speed - decel * (float(times[k]) - t0))


def rolling_angular_velocity(velocity: ArrayLike, radius: float) -> Vec:
    """omega = z x v / r for rolling without slipping on a horizontal floor."""
    out: Vec = np.cross(np.array([0.0, 0.0, 1.0]), np.asarray(velocity, dtype=np.float64)) / radius
    return out


def propagate_quaternion(q0: ArrayLike, omega: ArrayLike, dt: float) -> Vec:
    """q(t) = exp(omega dt / 2) (x) q0 for a constant world angular velocity (wxyz)."""
    w = np.asarray(omega, dtype=np.float64)
    angle = float(np.linalg.norm(w)) * dt
    q = np.asarray(q0, dtype=np.float64)
    if angle < 1e-15:
        return q.copy()
    axis = w / np.linalg.norm(w)
    half = 0.5 * angle
    r = np.concatenate([[math.cos(half)], math.sin(half) * axis])
    return _quat_mul(r, q)


def _quat_mul(a: Vec, b: Vec) -> Vec:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def energy_scale(
    masses: Sequence[float], v_minus: Sequence[Vec], v_plus: Sequence[Vec], gain: float
) -> float:
    """lambda with v_i+' = v_cm- + lambda (v_i+ - v_cm+) and E+' = (1 + gain) E-:
    lambda^2 = ((1 + gain) E- - E_cm) / E_rel+ (ADR 0004 §3.5). NaN if unreachable."""
    m = np.asarray(masses, dtype=np.float64)
    before = np.asarray(v_minus, dtype=np.float64)
    after = np.asarray(v_plus, dtype=np.float64)
    cm_before = m @ before / m.sum()
    cm_after = m @ after / m.sum()
    e_minus = 0.5 * float(m @ np.sum(before**2, axis=1))
    e_cm = 0.5 * float(m.sum()) * float(cm_before @ cm_before)
    e_rel = 0.5 * float(m @ np.sum((after - cm_after) ** 2, axis=1))
    square = ((1.0 + gain) * e_minus - e_cm) / e_rel
    return math.sqrt(square) if square > 0.0 and math.isfinite(square) else math.nan
