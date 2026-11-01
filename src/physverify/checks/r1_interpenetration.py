"""R1 interpenetration of object pairs and of each object with the ground."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.checks._base import CheckInputs, abstain, unscored
from physverify.checks._params import OcclusionParams, as_params
from physverify.physics.geometry import (
    Body,
    Plane,
    axis_rotation,
    characteristic_length,
    matrix_from_quaternion,
    quaternion_from_matrix,
)
from physverify.physics.measures.penetration import penetration_ratio
from physverify.physics.signed_distance import signed_distance
from physverify.physics.tracks import native_span, place_body, support_plane
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind
from physverify.schema.states import ObjectStateTable, PrimitiveKind

FloatArray = NDArray[np.float64]
# Horizontal axis used to orient a body whose quaternion is unknown (yaw is irrelevant for the
# rotation-symmetric distance bounds R1 needs; any fixed axis keeps results reproducible).
YAW_AXIS = np.array([1.0, 0.0, 0.0])
# Relative step of the central differences that propagate position and size sigmas to d.
JACOBIAN_STEP = 1e-6
SEPARATED_SIGMAS = 3.0
# A support surface is where a body rests, not where it bounces: an impact record lies at the
# body's bottom at one instant, at whatever depth the lift gave that instant.
SUPPORT_KINDS = frozenset({ContactKind.RESTING})


class InterpenetrationParams(OcclusionParams):
    pass


@dataclass(frozen=True)
class _Peak:
    row: int
    distance: float
    ratio: float
    assumed: bool


class InterpenetrationCheck:
    """R1 [pen_over_size]: max over rows of the penetration depth over the smaller L.

    The maximum of noisy per-row values is biased upward: with position or size noise a clean
    contact (true d = 0) still yields max_t max(0, -d_t) / L > 0, growing with the noise and the
    number of rows. ``sigma`` is the per-row sigma at the peak, not the spread of the maximum.
    The bias is not corrected here: tau is a conformal quantile of this same maximum over clean
    clips (``calibration.thresholds.fit_tau``), so the null distribution already contains it.
    """

    check_id: ClassVar[str] = "R1"
    requires: ClassVar[frozenset[str]] = frozenset()
    time_dependent: ClassVar[bool] = False
    magnitude_unit: ClassVar[str] = "pen_over_size"
    Params: ClassVar[type[BaseModel]] = InterpenetrationParams

    def __init__(self, params: BaseModel) -> None:
        self.params = as_params(InterpenetrationParams, params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        return [self._subject(inputs, i, other) for i, other in self._subjects(inputs)]

    def _subjects(self, inputs: CheckInputs) -> Iterator[tuple[int, int | Plane]]:
        n = inputs.states.n_objects
        for i in range(n):
            for j in range(i + 1, n):
                yield i, j
        if inputs.contacts is None:
            return
        for i in range(n):
            object_id = str(inputs.states.object_ids[i])
            plane = support_plane(object_id, inputs.contacts, SUPPORT_KINDS)
            if plane is not None:
                yield i, plane

    def _subject(self, inputs: CheckInputs, i: int, other: int | Plane) -> CheckResult:
        states = inputs.states
        ids = states.object_ids
        subject = (str(ids[i]), ENV_ID if isinstance(other, Plane) else str(ids[other]))
        ground = {"ground": "records" if isinstance(other, Plane) else "none"}
        peak = self._peak(states, i, other)
        if peak is None:
            reason = self._blocked_reason(states, i, other)
            return abstain(type(self), inputs, subject, reason, None, ground)
        fi = states.frame_index
        sigma = self._sigma(states, peak, i, other)
        details = {
            **ground,
            "distance": peak.distance,
            "orientation_assumed": peak.assumed,
            "frame": int(fi[peak.row]),
            "row": peak.row,
        }
        span = native_span(fi, peak.row, peak.row)
        return unscored(type(self), inputs, subject, peak.ratio, sigma, span, details)

    def _peak(self, states: ObjectStateTable, i: int, other: int | Plane) -> _Peak | None:
        peaks = []
        for row in range(states.t.shape[0]):
            placed = _place_pair(states, row, i, other)
            if placed is None:
                continue
            a, b, assumed = placed
            distance = signed_distance(a, b).distance
            ratio = penetration_ratio(distance, *_extents(states, row, i, other))
            peaks.append(_Peak(row, distance, ratio, assumed))
        if not peaks:
            return None
        return min(peaks, key=lambda p: (-p.ratio, p.distance, p.row))

    def _blocked_reason(
        self, states: ObjectStateTable, i: int, other: int | Plane
    ) -> AbstainReason:
        """OCCLUDED when every unplaceable body of every row is occluded, else MISSING_INPUT."""
        bodies = [i] if isinstance(other, Plane) else [i, other]
        for row in range(states.t.shape[0]):
            for body in bodies:
                if place_body(states, row, body, YAW_AXIS) is None and not (
                    float(states.occlusion[row, body]) >= self.params.occluded_at
                ):
                    return AbstainReason.MISSING_INPUT
        return AbstainReason.OCCLUDED

    def _sigma(self, states: ObjectStateTable, peak: _Peak, i: int, other: int | Plane) -> float:
        """Sigma of the ratio: first order in the position and size sigmas of both bodies, plus
        the yaw spread of each posed body; NaN when any of them is unknown."""
        bodies = [i] if isinstance(other, Plane) else [i, other]
        row = peak.row
        sigmas = np.concatenate(
            [
                np.concatenate([states.position_sigma[row, b], states.size_sigma[row, b]])
                for b in bodies
            ]
        )
        yaw_spread = _yaw_spread(states, peak, i, other, bodies)
        if not (bool(np.all(np.isfinite(sigmas))) and math.isfinite(yaw_spread)):
            return math.nan
        if bool(np.all(sigmas == 0.0)) and yaw_spread == 0.0:
            return 0.0
        gradient = _distance_gradient(states, row, i, other, bodies)
        sigma_d = float(np.sqrt(np.sum((gradient * sigmas) ** 2) + yaw_spread**2))
        if peak.ratio == 0.0 and peak.distance - SEPARATED_SIGMAS * sigma_d > 0.0:
            return 0.0
        lengths = [characteristic_length(states.size[row, b]) for b in bodies]
        return sigma_d / min(lengths)


def _place_pair(
    states: ObjectStateTable, row: int, i: int, other: int | Plane
) -> tuple[Body, Body, bool] | None:
    a = place_body(states, row, i, YAW_AXIS)
    if a is None:
        return None
    if isinstance(other, Plane):
        return a.solid, other, a.assumed
    b = place_body(states, row, other, YAW_AXIS)
    if b is None:
        return None
    return a.solid, b.solid, a.assumed or b.assumed


def _extents(
    states: ObjectStateTable, row: int, i: int, other: int | Plane
) -> tuple[FloatArray, FloatArray]:
    """Ground contacts normalize by the object's own L."""
    size_i = states.size[row, i]
    return size_i, size_i if isinstance(other, Plane) else states.size[row, other]


def _distance_gradient(
    states: ObjectStateTable, row: int, i: int, other: int | Plane, bodies: list[int]
) -> FloatArray:
    """Central differences of d at ``row`` over (position, size) of each body, in that order."""
    scale = min(characteristic_length(states.size[row, b]) for b in bodies)
    step = JACOBIAN_STEP * scale
    gradient = []
    for body in bodies:
        for field in ("position", "size"):
            for axis in range(3):
                plus = _distance(_nudged(states, field, (row, body, axis), step), row, i, other)
                minus = _distance(_nudged(states, field, (row, body, axis), -step), row, i, other)
                gradient.append((plus - minus) / (2.0 * step))
    return np.asarray(gradient, dtype=np.float64)


def _yaw_spread(
    states: ObjectStateTable, peak: _Peak, i: int, other: int | Plane, bodies: list[int]
) -> float:
    """Root sum of squares over posed non-sphere bodies of max |d(+-sigma) - d|, turning each by
    its orientation sigma about up = -gravity. Sigma points, not a Jacobian: at an aligned pose
    d is flat in yaw to first order. NaN when a posed body's sigma or the up axis is unknown."""
    total = 0.0
    for body in bodies:
        sigma = _orientation_sigma(states, peak.row, body)
        if sigma == 0.0:
            continue
        if not (math.isfinite(sigma) and bool(np.all(np.isfinite(states.gravity_dir)))):
            return math.nan
        turned = [
            _distance(_yawed(states, peak.row, body, angle), peak.row, i, other)
            for angle in (sigma, -sigma)
        ]
        total += max(abs(d - peak.distance) for d in turned) ** 2
    return math.sqrt(total)


def _orientation_sigma(states: ObjectStateTable, row: int, body: int) -> float:
    """Yaw sigma of a body placed with its own quaternion; 0 when R1 assumes the orientation
    (the assumption is reported as ``orientation_assumed``) or the body is a sphere."""
    posed = bool(np.all(np.isfinite(states.quaternion[row, body])))
    if not posed or PrimitiveKind(str(states.primitive[body])) is PrimitiveKind.SPHERE:
        return 0.0
    return float(states.orientation_sigma[row, body])


def _yawed(states: ObjectStateTable, row: int, body: int, angle: float) -> ObjectStateTable:
    turn = axis_rotation(-states.gravity_dir, angle)
    quaternions = np.array(states.quaternion, copy=True)
    rotation = turn @ matrix_from_quaternion(quaternions[row, body])
    quaternions[row, body] = quaternion_from_matrix(rotation)
    return replace(states, quaternion=quaternions)


def _nudged(
    states: ObjectStateTable, field: str, where: tuple[int, int, int], step: float
) -> ObjectStateTable:
    values = np.array(getattr(states, field), copy=True)
    values[where] += step
    changes: dict[str, Any] = {field: values}
    return replace(states, **changes)


def _distance(states: ObjectStateTable, row: int, i: int, other: int | Plane) -> float:
    placed = _place_pair(states, row, i, other)
    return math.nan if placed is None else signed_distance(placed[0], placed[1]).distance


PLUGINS = (InterpenetrationCheck,)
