"""Row-aware helpers for the checks: a table row is not a native frame (ADR 0006, D14).

A state table may be strided, so row k sits at native frame ``frame_index[k]`` and consecutive
rows are ``s = row_step(frame_index)`` native frames apart. Contact times are fractional native
frames; every window below widens them by the quantization margin ``q = s - 1``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from physverify.physics.finite_diff import second_difference, third_difference
from physverify.physics.geometry import (
    Plane,
    Pose,
    Primitive,
    Solid,
    Vec,
    assumed_upright_pose,
    characteristic_length,
    pose_from_quaternion,
)
from physverify.physics.kinematics import UNIFORM_STEP_RTOL, visible_runs
from physverify.physics.measures.r5_gravity import FallFit, fall_fit
from physverify.physics.uncertainty import robust_scale
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable
from physverify.schema.states import ObjectStateTable, PrimitiveKind

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
MIN_FIT_ROWS = 3  # fall_fit needs three rows with a known position
# Occlusion share from which an unseen object counts as hidden: the checks that read occlusion
# and the perceived gap estimate of LIFT (extraction/providers/_visibility.py) use the same value.
OCCLUDED_AT = 0.9
# Fewest usable third-difference stencils from which a track measures its own noise: the fall
# filter of R5 and the kinematic impacts of CONTACTS use the same value.
MIN_NOISE_STENCILS = 8


@dataclass(frozen=True, eq=False)
class PlacedBody:
    solid: Solid
    assumed: bool
    point_only: bool


@dataclass(frozen=True, eq=False)
class PolyFit:
    """y(t) = sum_k coef[k] (t - t0)^k per component; ``cov`` holds the coefficient covariance
    per component (None when an input sigma is unknown)."""

    t0: float
    coef: FloatArray  # (degree + 1, *shape)
    cov: FloatArray | None  # (*shape, degree + 1, degree + 1)

    @property
    def degree(self) -> int:
        return int(self.coef.shape[0]) - 1

    def value_at(self, t: float, derivative: int = 0) -> tuple[FloatArray, FloatArray]:
        """The ``derivative``-th time derivative of the fit at ``t`` and its sigma."""
        basis = _derivative_basis(self.degree, t - self.t0, derivative)
        value = np.asarray(np.tensordot(basis, self.coef, axes=1), dtype=np.float64)
        if self.cov is None:
            return value, np.full(value.shape, math.nan)
        variance = np.einsum("i,...ij,j->...", basis, self.cov, basis)
        return value, np.sqrt(np.maximum(variance, 0.0))


def _derivative_basis(degree: int, tau: float, derivative: int) -> FloatArray:
    """d^r/dt^r of (1, tau, ..., tau^degree): k! / (k - r)! tau^(k - r), zero for k < r."""
    return np.array(
        [
            math.perm(k, derivative) * tau ** (k - derivative) if k >= derivative else 0.0
            for k in range(degree + 1)
        ]
    )


FitSource = Literal["position", "velocity"]
# Polynomial degree of the per-side fit: between impacts the only force is gravity, so the
# position is a parabola and the velocity a line.
MOTION_DEGREE: Mapping[FitSource, int] = {"position": 2, "velocity": 1}


@dataclass(frozen=True, eq=False)
class MotionFit:
    """Constant-acceleration motion fitted on positions (parabola) or velocities (line)."""

    fit: PolyFit
    source: FitSource

    @property
    def _order(self) -> int:
        """Time derivative of the fitted quantity that gives the velocity."""
        return 1 if self.source == "position" else 0

    def velocity_at(self, t: float) -> tuple[FloatArray, FloatArray]:
        return self.fit.value_at(t, self._order)

    @property
    def acceleration(self) -> FloatArray:
        return self.fit.value_at(self.fit.t0, self._order + 1)[0]


def motion_fit(t: FloatArray, y: FloatArray, sigma: FloatArray, source: FitSource) -> MotionFit:
    """MotionFit of positions or velocities ``y`` (n, 3) sampled at times ``t``."""
    return MotionFit(poly_fit(t, y, sigma, MOTION_DEGREE[source]), source)


def row_step(frame_index: IntArray) -> int:
    """Uniform native step between consecutive rows (1 for a single row)."""
    steps = np.diff(np.asarray(frame_index, dtype=np.int64))
    if steps.size == 0:
        return 1
    step = int(steps[0])
    if step < 1 or bool(np.any(steps != step)):
        raise ValueError(f"frame_index needs one uniform step >= 1, got {np.unique(steps)}")
    return step


def check_time_axis(t: FloatArray, frame_index: IntArray, seconds_per_frame: float) -> None:
    """Raise unless every row time equals its native frame times the frame period (D14.2)."""
    if not (math.isfinite(seconds_per_frame) and seconds_per_frame > 0):
        raise ValueError(f"seconds_per_frame must be finite and > 0, got {seconds_per_frame}")
    times = np.asarray(t, dtype=np.float64)
    native = np.asarray(frame_index, dtype=np.int64) * seconds_per_frame
    if times.shape != native.shape:
        raise ValueError(f"t has shape {times.shape} but frame_index has {native.shape}")
    gap = float(np.max(np.abs(times - native), initial=0.0))
    if not gap <= UNIFORM_STEP_RTOL * seconds_per_frame:
        raise ValueError(f"t departs from frame_index * seconds_per_frame by {gap:.3g} s")


def records_of(object_id: str, contacts: ContactTable) -> list[ContactRecord]:
    return [r for r in contacts.records if object_id in (r.object_a, r.object_b)]


def contact_rows(record: ContactRecord, frame_index: IntArray, margin: int = 0) -> BoolArray:
    """Rows with floor(t_start) - m - q <= frame_index <= ceil(t_end) + m + q."""
    fi = np.asarray(frame_index, dtype=np.int64)
    widen = margin + row_step(fi) - 1
    first = math.floor(record.t_start) - widen
    last = math.ceil(record.t_end) + widen
    return np.asarray((fi >= first) & (fi <= last), dtype=np.bool_)


def flight_mask(
    object_id: str, contacts: ContactTable, frame_index: IntArray, margin: int = 1
) -> BoolArray:
    """Rows outside every contact window of the object."""
    fi = np.asarray(frame_index, dtype=np.int64)
    touching = np.zeros(fi.shape, dtype=np.bool_)
    for record in records_of(object_id, contacts):
        touching |= contact_rows(record, fi, margin)
    return ~touching


def impact_stencil_mask(object_id: str, contacts: ContactTable, frame_index: IntArray) -> BoolArray:
    """Rows whose third-difference stencil (rows k-1..k+2) spans an impact of the object.

    Row k is excluded when the closed interval [fi[k-1], fi[k+2]] meets [t_start - q, t_end + q].
    A bound on a sampled frame is treated as quantized: the event lies strictly within one step
    beyond it (perceived contacts snap to frames). Fractional ground-truth bounds keep A4's
    floor(t_start) - 1 <= k <= ceil(t_end) at the native rate; integer bounds exclude one more
    row on each side. Rows past the ends use virtual neighbours one or two steps outside.
    """
    fi = np.asarray(frame_index, dtype=np.int64)
    excluded = np.zeros(fi.shape, dtype=np.bool_)
    if fi.size == 0:
        return excluded
    step = row_step(fi)
    before = np.concatenate([[fi[0] - step], fi[:-1]])
    after = np.concatenate([fi, fi[-1] + step * np.arange(1, 3)])[2:]
    for record in records_of(object_id, contacts):
        if record.kind is ContactKind.IMPACT:
            q = step - 1
            excluded |= (before <= record.t_end + q) & (after >= record.t_start - q)
    return excluded


def noise_stencils(
    states: ObjectStateTable, contacts: ContactTable, index: int
) -> tuple[FloatArray, BoolArray]:
    """Third differences D3[j] of the object's positions (stencil rows j..j+3) and the stencils
    usable as noise: finite and not spanning an impact. D3[j] is row k = j + 1 of
    ``impact_stencil_mask``, whose stencil is rows k-1..k+2."""
    d3 = third_difference(states.position[:, index])
    if not d3.shape[0]:
        return d3, np.zeros(0, dtype=np.bool_)
    object_id = str(states.object_ids[index])
    impacts = impact_stencil_mask(object_id, contacts, states.frame_index)[1:-2]
    usable = np.all(np.isfinite(d3), axis=-1) & ~impacts
    return d3, np.asarray(usable, dtype=np.bool_)


def track_noise(
    states: ObjectStateTable, contacts: ContactTable, index: int, min_stencils: int
) -> FloatArray | None:
    """Per-axis white-noise sigma of the positions, robust_scale(D3) / sqrt(20); None with fewer
    than ``min_stencils`` usable stencils.

    Var(D3) = (1 + 9 + 9 + 1) sigma^2 for independent noise, and D3 vanishes under any constant
    acceleration, so the estimate does not depend on g. A drift slow enough to be quadratic over
    the stencil is invisible to it."""
    d3, usable = noise_stencils(states, contacts, index)
    if int(usable.sum()) < min_stencils:
        return None
    return np.array([robust_scale(d3[usable, axis]) for axis in range(d3.shape[1])]) / math.sqrt(
        20.0
    )


def segments(mask: BoolArray, min_len: int) -> list[tuple[int, int]]:
    """Maximal runs of True rows as [start, stop), keeping those with >= ``min_len`` rows."""
    return [(r.start, r.stop) for r in visible_runs(mask) if r.stop - r.start >= min_len]


def native_span(frame_index: IntArray, first_row: int, last_row: int) -> tuple[int, int]:
    """The only constructor of a result's ``frame_span``: native frames of two rows."""
    return int(frame_index[first_row]), int(frame_index[last_row])


def median_extents(states: ObjectStateTable, index: int) -> FloatArray:
    """Median extents of object ``index`` over rows with a finite size; NaN without any."""
    size = states.size[:, index]
    known = np.all(np.isfinite(size), axis=-1)
    if not bool(known.any()):
        return np.full(3, math.nan)
    return np.asarray(np.median(size[known], axis=0), dtype=np.float64)


def body_length(states: ObjectStateTable, index: int) -> float:
    """L_i = characteristic_length of the median extents over rows with a finite size."""
    extents = median_extents(states, index)
    return characteristic_length(extents) if bool(np.all(np.isfinite(extents))) else math.nan


def rest_mask(states: ObjectStateTable, index: int, rest_speed: float) -> BoolArray:
    """Rows with ||v|| <= rest_speed * L_i (rest_speed in body lengths per second)."""
    speed = np.linalg.norm(states.velocity[:, index], axis=-1)
    with np.errstate(invalid="ignore"):
        return np.asarray(speed <= rest_speed * body_length(states, index), dtype=np.bool_)


def split_at_jumps(
    states: ObjectStateTable, index: int, seg: tuple[int, int], gate: float
) -> list[tuple[int, int]]:
    """Cut the row segment where ||a_k|| > gate * median ||a|| (D12).

    A jump between rows j-1 and j inflates both a_{j-1} and a_j, so each run of flagged rows is
    cut at its middle and the two rows land on opposite sides.
    """
    lo, hi = seg
    accel = second_difference(states.position[lo:hi, index], states.t[lo:hi])
    norms = np.linalg.norm(accel, axis=-1)
    finite = np.isfinite(norms)
    if not bool(finite.any()):
        return [seg]
    flagged = finite & (norms > gate * float(np.median(norms[finite])))
    cuts = [lo + run.start + (run.stop - run.start + 1) // 2 for run in visible_runs(flagged)]
    bounds = [lo, *cuts, hi]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]


@dataclass(frozen=True, eq=False)
class Flight:
    """Rows [lo, hi) of object ``index`` in free flight; ``rows`` have a known position."""

    index: int
    lo: int
    hi: int
    rows: IntArray

    @property
    def present(self) -> int:
        return int(self.rows.size)


def find_flights(
    states: ObjectStateTable,
    contacts: ContactTable,
    index: int,
    rest_speed: float,
    jump_gate: float,
) -> list[Flight]:
    """Flights of object ``index`` with at least MIN_FIT_ROWS rows present, in time order."""
    object_id = str(states.object_ids[index])
    free = flight_mask(object_id, contacts, states.frame_index)
    free &= ~rest_mask(states, index, rest_speed)
    known = np.all(np.isfinite(states.position[:, index]), axis=-1)
    pieces = [
        piece
        for seg in segments(free, MIN_FIT_ROWS)
        for piece in split_at_jumps(states, index, seg, jump_gate)
    ]
    flights = [Flight(index, lo, hi, lo + np.flatnonzero(known[lo:hi])) for lo, hi in pieces]
    return [f for f in flights if f.present >= MIN_FIT_ROWS]


def fit_flight(states: ObjectStateTable, flight: Flight, offset: Any = 0.0) -> FallFit:
    """Parabola over the flight's rows against table time; ``offset`` perturbs the positions."""
    positions = states.position[flight.rows, flight.index] + offset
    return fall_fit(states.t[flight.rows], positions)


def flights_span(frame_index: IntArray, flights: Sequence[Flight]) -> tuple[int, int]:
    """Native frames from the first row of the earliest flight to the last of the latest."""
    return native_span(frame_index, min(f.lo for f in flights), max(f.hi - 1 for f in flights))


def place_body(states: ObjectStateTable, row: int, index: int, yaw_axis: Vec) -> PlacedBody | None:
    """Solid of object ``index`` at ``row``; None when it is hidden or its extents are unknown."""
    position, size = states.position[row, index], states.size[row, index]
    known = bool(np.all(np.isfinite(position)) and np.all(np.isfinite(size)) and np.all(size > 0))
    if not (bool(states.visible[row, index]) and known):
        return None
    kind = PrimitiveKind(str(states.primitive[index]))
    quaternion = states.quaternion[row, index]
    if bool(np.all(np.isfinite(quaternion))):
        pose = pose_from_quaternion(position, quaternion)
        return PlacedBody(Solid(Primitive(kind, size), pose), assumed=False, point_only=False)
    if bool(np.all(np.isfinite(states.gravity_dir))):
        pose = assumed_upright_pose(position, states.gravity_dir, yaw_axis)
        # Extents were measured along the table axes; re-express them along the assumed axes.
        local = np.sqrt((pose.rotation**2).T @ size**2)
        return PlacedBody(Solid(Primitive(kind, local), pose), assumed=True, point_only=False)
    diameter = characteristic_length(size)
    sphere = Primitive(PrimitiveKind.SPHERE, np.full(3, diameter))
    return PlacedBody(Solid(sphere, Pose(position, np.eye(3))), assumed=True, point_only=True)


def support_plane(object_id: str, contacts: ContactTable) -> Plane | None:
    """Environment surface under the object's ENV contacts; None without ENV records.

    ENV is always ``object_b``, so each record normal already points from the environment to
    the object. Points lie midway between the surfaces, hence offset = median(n.p - d/2).
    """
    env = [r for r in records_of(object_id, contacts) if r.object_b == ENV_ID]
    if not env:
        return None
    normal = np.median(np.stack([r.normal for r in env]), axis=0)
    length = float(np.linalg.norm(normal))
    if not length > 1e-9:
        return None
    normal = normal / length
    offsets = [r.points @ normal - _distances(r) / 2.0 for r in env]
    return Plane(normal=normal, offset=float(np.median(np.concatenate(offsets))))


def _distances(record: ContactRecord) -> FloatArray:
    if record.point_distances is None:
        return np.zeros(record.points.shape[0])
    return record.point_distances


def poly_fit(t: FloatArray, y: FloatArray, sigma: FloatArray, degree: int) -> PolyFit:
    """Least-squares polynomial of ``degree`` in t - t0 per component, t0 = mean(t), weights
    1/sigma^2.

    With any zero sigma the fit is unweighted and the covariance still propagates the given
    sigmas; with any NaN sigma the covariance is unknown (None).
    """
    times = np.asarray(t, dtype=np.float64)
    values = np.asarray(y, dtype=np.float64)
    if degree < 0 or times.ndim != 1 or values.shape[:1] != times.shape:
        raise ValueError("poly_fit needs degree >= 0 and y rows matching t")
    if np.unique(times).size <= degree:
        raise ValueError(f"a degree {degree} fit needs at least {degree + 1} distinct times")
    sigmas = np.broadcast_to(np.asarray(sigma, dtype=np.float64), values.shape)
    n, shape = times.size, values.shape[1:]
    flat_y, flat_s = values.reshape(n, -1), sigmas.reshape(n, -1)
    t0 = float(times.mean())
    design = np.vander(times - t0, degree + 1, increasing=True)
    known = bool(np.all(np.isfinite(flat_s)))
    gain = _fit_gains(design, flat_s)
    coef = np.einsum("mkn,nm->km", gain, flat_y)
    cov = (gain * (flat_s.T**2)[:, None, :]) @ gain.transpose(0, 2, 1) if known else None
    return PolyFit(
        t0=t0,
        coef=coef.reshape((degree + 1, *shape)),
        cov=None if cov is None else cov.reshape((*shape, degree + 1, degree + 1)),
    )


def _fit_gains(design: FloatArray, sigma: FloatArray) -> FloatArray:
    """(m, p, n) matrices G_j with coefficients_j = G_j y_j, one per column of ``sigma``, for
    an (n, p) design matrix."""
    usable = np.all(np.isfinite(sigma) & (sigma > 0), axis=0)
    safe = np.where(usable, sigma, 1.0)
    weights = np.where(usable, 1.0 / safe**2, 1.0).T
    weighted = design.T[None, :, :] * weights[:, None, :]
    gain: FloatArray = np.linalg.solve(weighted @ design, weighted)
    return gain
