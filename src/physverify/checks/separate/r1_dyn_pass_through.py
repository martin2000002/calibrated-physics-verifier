"""R1_dyn: a moving body passes through another one, reported as a row of its own."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs, abstain, unscored
from physverify.checks._params import CommonParams, as_params
from physverify.checks.separate._image import (
    FloatArray,
    camera_k,
    outline,
    overlapping,
    project,
    row_sigma,
)
from physverify.physics.geometry import matrix_from_quaternion, plane_basis
from physverify.physics.kinematics import visible_runs
from physverify.physics.tracks import native_span
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.states import ObjectStateTable, PrimitiveKind

BoolArray = NDArray[np.bool_]
_CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
_RIM = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)


class PassThroughParams(CommonParams):
    """``min_frames`` is k (native frames of one run of matched rows); the floor
    contact rows match within ``match_z`` sigma; each image rate is fitted on at least
    ``min_rate_rows`` rows."""

    min_frames: int = Field(default=3, ge=1)
    match_z: float = Field(default=1.96, gt=0.0)
    min_rate_rows: int = Field(default=3, ge=2)


@dataclass(frozen=True)
class ImageTrack:
    """One body on the image, per row: its silhouette (None where unusable), its image size
    s = f D / z (D the bounding diameter of ``size``), the image rows [low, high] its floor
    footprint spans, the pixel of its lowest point and the delta-method sigma of that row."""

    silhouette: list[FloatArray | None]
    size: FloatArray
    low: FloatArray
    high: FloatArray
    contact: FloatArray
    sigma: FloatArray

    @property
    def usable(self) -> BoolArray:
        return np.array([s is not None for s in self.silhouette], dtype=bool)


class PassThroughCheck:
    """R1_dyn [frames]: s = n (1 - impulse), n the native frames of the longest run in which the
    pair's silhouettes overlap and their floor shadows span the same image rows (within
    ``match_z`` sigma); s = 0 when n < ``min_frames`` (k and 1.96 sigma fixed a priori).

    Every quantity is read on the image plane, so it does not depend on the depth scale. The
    silhouettes are the projected primitives (the occluding circle of a sphere, the hull of the
    8 box corners otherwise): ``CheckInputs`` carries no masks. On a floor seen from above, the
    image row of a floor point orders its depth. Two bodies that overlap in 3D have floor
    shadows that overlap, so their footprints share image rows: the row match is the depth half
    of the test, the silhouette overlap the image half. The impulse is read from
    D = ((u - c_x) / s, (v - c_y) / s, f / s) of each body's lowest point (u, v): D is the
    position in body lengths, linear in time at constant velocity, and its rate uses only the
    image-size change and the contact row, never a metric depth velocity. With rates fitted by
    least squares on the usable rows before the run and from its first row on,
    impulse = clip(max_b |D'_b,after - D'_b,before| / |D'_m,before|, 0, 1), m the body with the
    faster rate before (the mover).

    The row sigma propagates ``position_sigma`` of the lowest points; it ignores size and
    orientation errors. The result's sigma is 0 for exact inputs and NaN otherwise: the row's
    out-of-fold calibration supplies it."""

    check_id: ClassVar[str] = "R1_dyn"
    version: ClassVar[str] = "1"
    requires: ClassVar[frozenset[str]] = frozenset({"camera"})
    time_dependent: ClassVar[bool] = True
    magnitude_unit: ClassVar[str] = "frames"
    Params: ClassVar[type[BaseModel]] = PassThroughParams

    def __init__(self, params: BaseModel) -> None:
        self.params = as_params(PassThroughParams, params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        states = inputs.states
        ids = [str(x) for x in states.object_ids]
        pairs = [(i, j) for i in range(states.n_objects) for j in range(i + 1, states.n_objects)]
        k = camera_k(inputs)
        if k is None:
            missing = AbstainReason.MISSING_INPUT
            return [
                abstain(type(self), inputs, (ids[i], ids[j]), missing, None, {}) for i, j in pairs
            ]
        tracks = [image_track(k, states, i) for i in range(states.n_objects)]
        return [self._pair(inputs, k, tracks[i], tracks[j], (ids[i], ids[j])) for i, j in pairs]

    def _pair(
        self,
        inputs: CheckInputs,
        k: FloatArray,
        a: ImageTrack,
        b: ImageTrack,
        subject: tuple[str, str],
    ) -> CheckResult:
        states, params = inputs.states, self.params
        usable = a.usable & b.usable
        overlap = np.array(
            [
                sa is not None and sb is not None and overlapping(sa, sb)
                for sa, sb in zip(a.silhouette, b.silhouette, strict=True)
            ],
            dtype=bool,
        )
        gap = np.maximum(a.low - b.high, b.low - a.high)
        band = params.match_z * np.hypot(a.sigma, b.sigma)
        if not bool(np.all(np.isfinite(band[overlap]))):
            reason = AbstainReason.UNKNOWN_UNCERTAINTY
            return abstain(type(self), inputs, subject, reason, None, {})
        sigma = 0.0 if bool(np.all(band[overlap] == 0.0)) else math.nan
        matched = overlap & (gap <= band)
        run = _longest_run(states.frame_index, matched)
        details: dict[str, Any] = {
            "overlap_rows": int(overlap.sum()),
            "matched_rows": int(matched.sum()),
            "matched_frames": 0 if run is None else run[2],
        }
        if run is None or run[2] < params.min_frames:
            return unscored(type(self), inputs, subject, 0.0, sigma, None, details)
        start, stop, frames = run
        span = native_span(states.frame_index, start, stop - 1)
        before = np.flatnonzero(usable[:start])
        after = start + np.flatnonzero(usable[start:])
        if before.size < params.min_rate_rows or after.size < params.min_rate_rows:
            reason = AbstainReason.TOO_FEW_FRAMES
            return abstain(type(self), inputs, subject, reason, span, details)
        rates = [
            (_rate(states.t, d, before), _rate(states.t, d, after))
            for d in (_descriptor(k, a), _descriptor(k, b))
        ]
        speeds = [float(np.linalg.norm(pre)) for pre, _ in rates]
        mover = int(np.argmax(speeds))
        reference = speeds[mover]
        if not (math.isfinite(reference) and reference > 0.0):
            reason = AbstainReason.ILL_CONDITIONED
            return abstain(type(self), inputs, subject, reason, span, details)
        changes = [float(np.linalg.norm(post - pre)) / reference for pre, post in rates]
        impulse = min(1.0, max(changes))
        details |= {
            "mover": subject[mover],
            "impulse": impulse,
            "rate_change": changes[mover],
            "other_rate_change": changes[1 - mover],
            "first_matched_frame": int(states.frame_index[start]),
        }
        magnitude = frames * (1.0 - impulse)
        return unscored(type(self), inputs, subject, magnitude, sigma, span, details)


def image_track(k: FloatArray, states: ObjectStateTable, index: int) -> ImageTrack:
    gravity = states.gravity_dir / np.linalg.norm(states.gravity_dir)
    focal, n = float(k[0, 0]), states.n_frames
    silhouette: list[FloatArray | None] = [None] * n
    contact = np.full((n, 2), math.nan)
    size, low, high, sigma = (np.full(n, math.nan) for _ in range(4))
    for t in range(n):
        position, extents = states.position[t, index], states.size[t, index]
        finite = bool(np.all(np.isfinite(position)) and np.all(np.isfinite(extents)))
        if not (finite and position[2] > 0 and states.in_frame[t, index]):
            continue
        surface, floor = _surface_and_footprint(states, index, t, gravity)
        silhouette[t] = outline(k, surface)
        size[t] = focal * float(np.linalg.norm(extents)) / float(position[2])
        rows = project(k, floor)[:, 1]
        low[t], high[t] = float(rows.min()), float(rows.max())
        lowest = floor.mean(axis=0)
        contact[t] = project(k, lowest)
        sigma[t] = row_sigma(k, lowest, states.position_sigma[t, index])
    return ImageTrack(silhouette, size, low, high, contact, sigma)


def _surface_and_footprint(
    states: ObjectStateTable, index: int, t: int, gravity: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Points whose projected hull is the body's silhouette, and its floor footprint (its shadow
    along gravity on the plane of its lowest point, centred on that point): for a sphere its
    occluding circle (centre c (1 - R^2/|c|^2), radius R sqrt(1 - R^2/|c|^2), normal to the ray)
    and a floor circle of radius R; otherwise the box corners and those corners flattened onto
    the plane of the lowest one (axes assumed when the quaternion is unknown)."""
    position, extents = states.position[t, index], states.size[t, index]
    if str(states.primitive[index]) == PrimitiveKind.SPHERE.value:
        radius = float(np.max(extents)) / 2.0
        ratio = min(1.0, radius**2 / float(position @ position))
        u, v = plane_basis(position)
        ring = np.outer(np.cos(_RIM), u) + np.outer(np.sin(_RIM), v)
        rim = position * (1.0 - ratio) + radius * math.sqrt(1.0 - ratio) * ring
        floor_u, floor_v = plane_basis(gravity)
        shadow = np.outer(np.cos(_RIM), floor_u) + np.outer(np.sin(_RIM), floor_v)
        return rim, position + radius * (gravity + shadow)
    quaternion = states.quaternion[t, index]
    rotation = matrix_from_quaternion(quaternion) if np.all(np.isfinite(quaternion)) else np.eye(3)
    corners = position + (_CORNERS * extents / 2.0) @ rotation.T
    heights = (corners - position) @ gravity
    return corners, corners + np.outer(float(heights.max()) - heights, gravity)


def _descriptor(k: FloatArray, track: ImageTrack) -> FloatArray:
    focal, cx, cy = float(k[0, 0]), float(k[0, 2]), float(k[1, 2])
    s = track.size
    columns = ((track.contact[:, 0] - cx) / s, (track.contact[:, 1] - cy) / s, focal / s)
    return np.stack(columns, axis=1)


def _rate(t: FloatArray, values: FloatArray, rows: NDArray[np.int64]) -> FloatArray:
    """Least-squares slope (per second) of ``values`` over ``rows``."""
    times = t[rows] - float(np.mean(t[rows]))
    centred = values[rows] - values[rows].mean(axis=0)
    out: FloatArray = times @ centred / float(times @ times)
    return out


def _longest_run(frame_index: NDArray[np.int64], mask: BoolArray) -> tuple[int, int, int] | None:
    """(start, stop, native frames) of the longest run of True rows; None without one."""
    runs = [(r.start, r.stop) for r in visible_runs(mask)]
    if not runs:
        return None
    start, stop = max(runs, key=lambda r: int(frame_index[r[1] - 1] - frame_index[r[0]]))
    return start, stop, int(frame_index[stop - 1] - frame_index[start]) + 1


PLUGINS = (PassThroughCheck,)
