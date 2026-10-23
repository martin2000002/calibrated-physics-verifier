"""R5b gravity value: the fitted free-fall acceleration in meters per second squared against g0, for
spheres only (the same restriction as the synthetic label) and only with a scale that does not come
from the fall itself (a scale fitted on g would make the test circular)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs
from physverify.checks._falls import find_falls
from physverify.checks._params import FallParams
from physverify.checks._results import abstain, unscored
from physverify.physics.flights import Flight, fit_flight, flights_span
from physverify.physics.measures.gravity_fit import gravity_rel_error
from physverify.physics.tracks import MIN_FIT_ROWS
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.constants import STANDARD_GRAVITY
from physverify.schema.contacts import ContactTable
from physverify.schema.scale import EffectiveScale, ScaleSource
from physverify.schema.states import ObjectStateTable, PrimitiveKind

SUBJECT: tuple[str, ...] = ()


class GravityValueParams(FallParams):
    min_flight_frames: int = Field(default=4, ge=MIN_FIT_ROWS)
    primitives: tuple[PrimitiveKind, ...] = (PrimitiveKind.SPHERE,)


class GravityValueCheck:
    """R5b [g_rel_error]: median over the clip's flights of | |a| mpu - g0 | / g0, with |a| from
    the parabola fitted against table time, so it does not depend on the row step."""

    check_id: ClassVar[str] = "R5b"
    requires: ClassVar[frozenset[str]] = frozenset({"contacts"})
    time_dependent: ClassVar[bool] = True
    magnitude_unit: ClassVar[str] = "g_rel_error"
    Params: ClassVar[type[BaseModel]] = GravityValueParams

    def __init__(self, params: BaseModel) -> None:
        self.params = GravityValueParams.model_validate(params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        if inputs.contacts is None:
            raise ValueError(f"{self.check_id} requires contacts")
        states = inputs.states
        flights = self._flights(states, inputs.contacts)
        if not flights:
            return []
        span = flights_span(states.frame_index, flights)
        details: dict[str, Any] = {
            "time_source": inputs.clip.time_source.value,
            "scale_source": None if inputs.scale is None else inputs.scale.source.value,
            "flights": len(flights),
        }
        scale = inputs.scale
        if scale is None or scale.source is ScaleSource.FREE_FALL:
            reason = (
                AbstainReason.SCALE_UNAVAILABLE
                if scale is None
                else AbstainReason.SCALE_FROM_GRAVITY
            )
            return [abstain(type(self), inputs, SUBJECT, reason, span, details)]
        long = [f for f in flights if f.present >= self.params.min_flight_frames]
        if not long:
            details["min_flight_frames"] = self.params.min_flight_frames
            reason = AbstainReason.TOO_FEW_FRAMES
            return [abstain(type(self), inputs, SUBJECT, reason, span, details)]
        return [self._scored(inputs, long, scale, details)]

    def _flights(self, states: ObjectStateTable, contacts: ContactTable) -> list[Flight]:
        kinds = {kind.value for kind in self.params.primitives}
        return [
            flight
            for i in range(states.n_objects)
            if str(states.primitive[i]) in kinds
            for flight in find_falls(states, contacts, i, self.params)
        ]

    def _scored(
        self,
        inputs: CheckInputs,
        flights: Sequence[Flight],
        scale: EffectiveScale,
        details: dict[str, Any],
    ) -> CheckResult:
        states = inputs.states
        mpu = scale.meters_per_unit
        accels = [fit_flight(states, f).a for f in flights]
        errors = [gravity_rel_error(float(np.linalg.norm(a)), mpu) for a in accels]
        middle = _middle(errors)
        magnitude = float(np.mean([errors[m] for m in middle]))
        sigma = _sigma(states, [flights[m] for m in middle], [accels[m] for m in middle], scale)
        span = flights_span(states.frame_index, flights)
        details = {**details, "used_flights": len(flights), "flight_errors": errors}
        return unscored(type(self), inputs, SUBJECT, magnitude, sigma, span, details)


def _middle(values: Sequence[float]) -> list[int]:
    """Indices whose mean is the median: one for an odd count, the two central for an even."""
    order = sorted(range(len(values)), key=lambda n: values[n])
    half = len(order) // 2
    return order[half : half + 1] if len(order) % 2 else order[half - 1 : half + 1]


def _sigma(
    states: ObjectStateTable,
    flights: Sequence[Flight],
    accels: Sequence[Any],
    scale: EffectiveScale,
) -> float:
    """First order on the mean of the central flights' errors e_f = |x_f - 1|, x_f = |a_f| mpu/g0:
    the position terms are independent per flight, the scale term (dx/x = ds/s) is shared.
    With one flight it is x sqrt((sigma_|a| / |a|)^2 + rel_sigma^2)."""
    weight = 1.0 / len(flights)
    per_unit = scale.meters_per_unit / STANDARD_GRAVITY
    ratios = [per_unit * float(np.linalg.norm(a)) for a in accels]
    position = sum(
        (per_unit * _accel_norm_sigma(states, f, a)) ** 2
        for f, a in zip(flights, accels, strict=True)
    )
    shared = sum(math.copysign(x, x - 1.0) for x in ratios) * scale.rel_sigma
    return weight * math.sqrt(position + shared**2)


def _accel_norm_sigma(states: ObjectStateTable, flight: Flight, accel: Any) -> float:
    """sigma_|a| = sqrt(sum (a_k / |a|)^2 sigma_a,k^2), sigma_a,k = sigma_p,k sqrt([(D^T D)^-1]_33)
    with D = [1, tau, tau^2 / 2], tau = t - t0 and sigma_p,k the median position sigma."""
    tau = states.t[flight.rows] - states.t[flight.rows[0]]
    design = np.column_stack([np.ones_like(tau), tau, 0.5 * tau**2])
    gain = math.sqrt(float(np.linalg.inv(design.T @ design)[2, 2]))
    sigma_p = np.median(states.position_sigma[flight.rows, flight.index], axis=0)
    sigma_a = gain * sigma_p
    norm = float(np.linalg.norm(accel))
    if norm == 0.0:
        return math.sqrt(float(np.mean(sigma_a**2)))
    return float(np.linalg.norm(np.asarray(accel) / norm * sigma_a))


PLUGINS = (GravityValueCheck,)
