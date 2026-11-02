"""R5b unsupported: R5b over the flights in which the body leaves its scene plane, so a stretch
the contacts left unmarked while the body rolls or slides on that plane is not read as a fall."""

from __future__ import annotations

import math
from typing import Any, ClassVar

from physverify.checks._base import CheckInputs, abstain
from physverify.checks.r1_interpenetration import YAW_AXIS
from physverify.checks.r5b_gravity_value import SUBJECT, GravityValueCheck
from physverify.physics.flights import Flight, flights_span
from physverify.physics.geometry import Plane
from physverify.physics.signed_distance import signed_distance
from physverify.physics.tracks import body_length, place_body, support_plane
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.contacts import ContactTable
from physverify.schema.states import ObjectStateTable


class UnsupportedGravityValueCheck(GravityValueCheck):
    """R5b unsupported [g_rel_error]: R5b's flights, each kept only when on some row the body's
    lowest point rises above its scene plane by more than K sigma_gap, with K sigma_gap / L =
    ``inputs.support_gap_bound`` (the gap gate of ADR 0032, the same z and sigma_gap). The plane
    of object n is the lowest environment surface under the other objects' ENV contacts, never
    n's own, as the scene contacts build it; a flight without such a plane, or without a placed
    row, is kept. The fitted acceleration is not read, so a body gliding above the plane with no
    acceleration is still scored. A clip whose every flight is supported has no result, as a clip
    without flights; with an unknown bound a clip with flights abstains (UNKNOWN_UNCERTAINTY)."""

    variant: ClassVar[str] = "unsupported"
    version: ClassVar[str] = "1"

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        if inputs.contacts is None or math.isfinite(inputs.support_gap_bound):
            return super().evaluate(inputs)
        flights = super()._flights(inputs)
        if not flights:
            return []
        span = flights_span(inputs.states.frame_index, flights)
        details: dict[str, Any] = {"flights": len(flights), "support_gap_bound": None}
        reason = AbstainReason.UNKNOWN_UNCERTAINTY
        return [abstain(type(self), inputs, SUBJECT, reason, span, details)]

    def _falls(self, inputs: CheckInputs, index: int) -> list[Flight]:
        falls = super()._falls(inputs, index)
        assert inputs.contacts is not None
        plane = scene_plane(inputs.states, inputs.contacts, index)
        if plane is None:
            return falls
        bound = inputs.support_gap_bound * body_length(inputs.states, index)
        return [f for f in falls if not supported(inputs.states, f, plane, bound)]


def scene_plane(states: ObjectStateTable, contacts: ContactTable, index: int) -> Plane | None:
    """The lowest of the other objects' environment planes (``support_plane``); None when no
    other object has an ENV contact."""
    others = (str(states.object_ids[m]) for m in range(states.n_objects) if m != index)
    planes = [p for object_id in others if (p := support_plane(object_id, contacts)) is not None]
    return min(planes, key=lambda p: p.offset, default=None)


def supported(states: ObjectStateTable, flight: Flight, plane: Plane, bound: float) -> bool:
    """True when every placed row of the flight lies within ``bound`` of ``plane`` (signed
    distance of the body's lowest point); False without a placed row or with a NaN bound."""
    gaps = [
        signed_distance(placed.solid, plane).distance
        for row in flight.rows
        if (placed := place_body(states, int(row), flight.index, YAW_AXIS)) is not None
    ]
    return bool(gaps) and max(gaps) <= bound


PLUGINS = (UnsupportedGravityValueCheck,)
