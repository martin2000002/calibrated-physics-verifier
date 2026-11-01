"""R2 footprint: the centre of mass of a body resting on other bodies against the outline of
those bodies, with the contact records used only to say who rests on whom."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs, abstain, unscored
from physverify.checks._params import as_params
from physverify.checks._support_graph import (
    Event,
    agents_in,
    components,
    reaches_ground,
    resting_mask,
    runs,
)
from physverify.checks.r2_static_equilibrium import FRICTION_RANGE, EquilibriumParams
from physverify.physics.footprint import outline_points, support_order
from physverify.physics.geometry import CYLINDER_SIDES, characteristic_length
from physverify.physics.measures.support_overhang import overhang_fraction
from physverify.physics.row_time import rows_in
from physverify.physics.tracks import median_extents, native_span
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel
from physverify.schema.constants import ENV_ID
from physverify.schema.error_profile import SUPPORT_OFFSET_REL
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable, PrimitiveKind

FloatArray = NDArray[np.float64]

SUPPORT_MAX_TILT_DEG = math.degrees(math.atan(FRICTION_RANGE[1]))
"""45 degrees: a contact plane steeper than atan(mu_max) cannot hold a body's weight by friction,
so a record whose normal leans further from the vertical is no support."""
GAP_OVER_LENGTH = 0.5
"""A body whose bottom lies more than half its own length from the top of its support (along
gravity) does not touch it in the perceived geometry: the record and the states disagree."""


class FootprintParams(EquilibriumParams):
    """v1's fields (rest, agents, support path, gravity sigma) plus the support geometry; the
    linear-program fields are kept so a shared ``[checks.R2]`` table stays valid."""

    support_max_tilt_deg: float = Field(default=SUPPORT_MAX_TILT_DEG, gt=0.0, lt=90.0)
    gap_over_length: float = Field(default=GAP_OVER_LENGTH, gt=0.0)
    rim_points: int = Field(default=CYLINDER_SIDES, ge=3)


@dataclass(frozen=True)
class _Body:
    """Pose of one body at the event row: centre, median extents, quaternion and shape."""

    center: FloatArray
    extents: FloatArray
    quaternion: FloatArray
    shape: str

    @property
    def known(self) -> bool:
        values = (self.center, self.extents, self.quaternion)
        return all(bool(np.all(np.isfinite(v))) for v in values) and bool(np.all(self.extents > 0))


class SupportFootprintCheck:
    """R2 footprint [overhang_over_size]: max over the bodies resting on other bodies (and on no
    environment surface) of the signed horizontal distance from the body's centroid to the hull of
    its supports' outlines (box corners, cylinder rims), over its own length; > 0 is outside.

    j supports i when a RESTING record links them at the event row with its normal within
    ``support_max_tilt_deg`` of the vertical, j below. No masses and no linear program: each body
    is held against its own supports only. sigma is the measured error of that relative offset
    (``support_offset_rel`` of the LIFT provider in the error profile); 0 for the oracle."""

    check_id: ClassVar[str] = "R2"
    variant: ClassVar[str] = "footprint"
    requires: ClassVar[frozenset[str]] = frozenset({"contacts"})
    time_dependent: ClassVar[bool] = False
    magnitude_unit: ClassVar[str] = "overhang_over_size"
    Params: ClassVar[type[BaseModel]] = FootprintParams

    def __init__(self, params: BaseModel) -> None:
        self.params = as_params(FootprintParams, params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        states, contacts = inputs.states, inputs.contacts
        if contacts is None:
            subject = tuple(sorted(str(i) for i in states.object_ids))
            details = {"missing": "contacts"}
            reason = AbstainReason.MISSING_INPUT
            return [abstain(type(self), inputs, subject, reason, None, details)]
        min_rows = rows_in(self.params.min_rest_s, inputs.row_seconds)
        resting = resting_mask(states, min_rows, self.params.rest_speed)
        found = (
            self._event(inputs, event)
            for start, stop in runs(resting)
            for event in components(states, contacts, resting, start, stop)
        )
        return [result for result in found if result is not None]

    def _event(self, inputs: CheckInputs, event: Event) -> CheckResult | None:
        states = inputs.states
        subject = tuple(str(states.object_ids[i]) for i in event.bodies)
        span = native_span(states.frame_index, event.start, event.stop - 1)
        details: dict[str, Any] = {
            "rest_rows": [event.start, event.stop - 1],
            "row": event.row,
            "frame": int(states.frame_index[event.row]),
        }
        reason = self._event_reason(inputs, event, details)
        if reason is not None:
            return abstain(type(self), inputs, subject, reason, span, details)
        borne = self._borne(states, event)
        if not borne:
            return None
        details["supports"] = {
            str(states.object_ids[i]): [str(states.object_ids[j]) for j in held]
            for i, held in borne.items()
        }
        reason = self._geometry_reason(states, event.row, borne, details)
        if reason is not None:
            return abstain(type(self), inputs, subject, reason, span, details)
        overhangs = {
            str(states.object_ids[i]): self._overhang(states, event.row, i, held)
            for i, held in borne.items()
        }
        details["overhang"] = overhangs
        sigma = self._sigma(inputs, details)
        if not math.isfinite(sigma):
            reason = AbstainReason.UNKNOWN_UNCERTAINTY
            return abstain(type(self), inputs, subject, reason, span, details)
        return unscored(type(self), inputs, subject, max(overhangs.values()), sigma, span, details)

    def _event_reason(
        self, inputs: CheckInputs, event: Event, details: dict[str, Any]
    ) -> AbstainReason | None:
        """v1's reasons before any geometry: gravity, centroids, agents, a path to the ground."""
        states = inputs.states
        if not bool(np.all(np.isfinite(states.gravity_dir))):
            details["missing"] = "gravity_dir"
            return AbstainReason.MISSING_INPUT
        if self.params.sample_gravity and not math.isfinite(states.gravity_dir_sigma):
            details["missing"] = "gravity_dir_sigma"
            return AbstainReason.UNKNOWN_UNCERTAINTY
        if not bool(np.all(np.isfinite(states.position[event.row, list(event.bodies)]))):
            details["missing"] = "position"
            return AbstainReason.MISSING_INPUT
        agents = agents_in(states, event, self.params.agent_labels)
        if agents:
            details["agents"] = agents
            return AbstainReason.OUT_OF_SCOPE
        perceived = inputs.input_level is not InputLevel.ORACLE
        if self.params.require_support_path and perceived and not reaches_ground(event):
            details["missing"] = "support_path"
            return AbstainReason.NO_SUPPORT_PATH
        return None

    def _borne(self, states: ObjectStateTable, event: Event) -> dict[int, list[int]]:
        """Bodies with no environment support -> the bodies under them (sorted by index)."""
        up = -np.asarray(states.gravity_dir, dtype=np.float64)
        on_ground = {r.object_a for r in event.support if r.object_b == ENV_ID}
        held: dict[int, set[int]] = {}
        for record in event.support:
            if record.object_b == ENV_ID:
                continue
            tilt = self.params.support_max_tilt_deg
            order = support_order(record.object_a, record.object_b, record.normal, up, tilt)
            if order is not None and order[0] not in on_ground:
                upper, lower = (states.object_index(name) for name in order)
                held.setdefault(upper, set()).add(lower)
        return {i: sorted(js) for i, js in sorted(held.items())}

    def _geometry_reason(
        self,
        states: ObjectStateTable,
        row: int,
        borne: dict[int, list[int]],
        details: dict[str, Any],
    ) -> AbstainReason | None:
        """Unknown poses, a point support (a sphere) or a support the body does not reach."""
        down, rim = _unit(states.gravity_dir), self.params.rim_points
        for i, held in borne.items():
            upper = _body(states, row, i)
            for j in [i, *held]:
                if not _body(states, row, j).known:
                    details["missing"] = f"pose:{states.object_ids[j]}"
                    return AbstainReason.MISSING_INPUT
            for j in held:
                lower = _body(states, row, j)
                if lower.shape == PrimitiveKind.SPHERE.value:
                    details["point_support"] = str(states.object_ids[j])
                    return AbstainReason.CONTACT_GEOMETRY_UNKNOWN
                gap = _lowest(upper, down, rim) - _highest(lower, down, rim)
                length = characteristic_length(upper.extents)
                if abs(gap) > self.params.gap_over_length * length:
                    details["support_gap_over_length"] = gap / length
                    return AbstainReason.SUPPORT_INCONSISTENT
        return None

    def _overhang(self, states: ObjectStateTable, row: int, index: int, held: list[int]) -> float:
        rim = self.params.rim_points
        outline = np.concatenate([_outline(_body(states, row, j), rim) for j in held])
        upper = _body(states, row, index)
        return overhang_fraction(upper.center, outline, states.gravity_dir, upper.extents)

    def _sigma(self, inputs: CheckInputs, details: dict[str, Any]) -> float:
        if inputs.input_level is InputLevel.ORACLE:
            return 0.0
        entry = inputs.stage_error(Stage.LIFT, SUPPORT_OFFSET_REL)
        if entry is None:
            details["missing"] = SUPPORT_OFFSET_REL
            return math.nan
        details["sigma_source"] = f"{entry.stage.value}/{entry.provider} n={entry.n}"
        return entry.sigma


def _unit(vector: Any) -> FloatArray:
    array = np.asarray(vector, dtype=np.float64)
    unit: FloatArray = array / np.linalg.norm(array)
    return unit


def _body(states: ObjectStateTable, row: int, index: int) -> _Body:
    return _Body(
        center=np.asarray(states.position[row, index], dtype=np.float64),
        extents=median_extents(states, index),
        quaternion=np.asarray(states.quaternion[row, index], dtype=np.float64),
        shape=str(states.primitive[index]),
    )


def _outline(body: _Body, rim: int) -> FloatArray:
    return outline_points(body.center, body.extents, body.quaternion, body.shape, rim)


def _heights(body: _Body, down: FloatArray, rim: int) -> FloatArray:
    """Heights (along -gravity) of the points that bound the body vertically."""
    if body.shape == PrimitiveKind.SPHERE.value:
        radius = float(body.extents[0]) / 2.0
        centre = -float(body.center @ down)
        return np.array([centre - radius, centre + radius])
    heights: FloatArray = -(_outline(body, rim) @ down)
    return heights


def _lowest(body: _Body, down: FloatArray, rim: int) -> float:
    return float(np.min(_heights(body, down, rim)))


def _highest(body: _Body, down: FloatArray, rim: int) -> float:
    return float(np.max(_heights(body, down, rim)))


PLUGINS = (SupportFootprintCheck,)
