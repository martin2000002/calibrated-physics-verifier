"""R5b line of sight: g along the perceived gravity fitted per flight by generalized least squares
on the covariance a monocular lift implies, instead of |a| of the per-axis parabola."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs, abstain, unscored
from physverify.checks.r5b_gravity_value import (
    SUBJECT,
    GravityValueCheck,
    GravityValueParams,
    mean_error_sigma,
    median_indices,
)
from physverify.checks.variants._line_of_sight import known_gravity, window_covariance
from physverify.checks.variants._whitened import EXACT_SIGMA_LENGTHS, floored_sigma
from physverify.physics.flights import Flight, flights_span
from physverify.physics.measures.vector_fall import BallisticFallFit, gls_ballistic_fit
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.constants import STANDARD_GRAVITY
from physverify.schema.scale import EffectiveScale
from physverify.schema.states import ObjectStateTable


class LineOfSightGravityValueParams(GravityValueParams):
    exact_sigma_lengths: float = Field(default=EXACT_SIGMA_LENGTHS, gt=0.0)


class LineOfSightGravityValueCheck(GravityValueCheck):
    """R5b los_ballistic [g_rel_error]: R5b's flights, each fitted with the ballistic GLS
    p0 + v0 tau + g_s g_dir tau^2 / 2 (seven unknowns, g_dir the perceived gravity as given)
    under Sigma_t = r_d^2 p p^T + diag(rest) built from R5a whitened's floored sigma, and
    x_f = g_s mpu / g0. The magnitude is the median of |x_f - 1| over the flights with a fit and
    its sigma ``mean_error_sigma`` with var(x_f) from the fit's own variance of g_s. A flight
    whose sigma is unknown is fitted unweighted and leaves the sigma unknown. Without the gravity
    direction a clip with flights abstains (UNKNOWN_UNCERTAINTY); with no flight fitted, as
    TOO_FEW_FRAMES."""

    variant: ClassVar[str] = "los_ballistic"
    version: ClassVar[str] = "1"
    Params: ClassVar[type[BaseModel]] = LineOfSightGravityValueParams
    params: LineOfSightGravityValueParams

    def __init__(self, params: BaseModel) -> None:
        self.params = LineOfSightGravityValueParams.model_validate(params)

    def _scored(
        self,
        inputs: CheckInputs,
        flights: Sequence[Flight],
        scale: EffectiveScale,
        details: dict[str, Any],
    ) -> CheckResult:
        states = inputs.states
        span = flights_span(states.frame_index, flights)
        if not known_gravity(states):
            reason = AbstainReason.UNKNOWN_UNCERTAINTY
            return abstain(type(self), inputs, SUBJECT, reason, span, details)
        fits = [fit for f in flights if (fit := self._flight_g(states, f)) is not None]
        if not fits:
            reason = AbstainReason.TOO_FEW_FRAMES
            return abstain(type(self), inputs, SUBJECT, reason, span, details)
        per_unit = scale.meters_per_unit / STANDARD_GRAVITY
        ratios = [per_unit * fit.g for fit in fits]
        errors = [abs(x - 1.0) for x in ratios]
        middle = median_indices(errors)
        magnitude = float(np.mean([errors[m] for m in middle]))
        variances = [per_unit**2 * fits[m].g_var for m in middle]
        sigma = mean_error_sigma(variances, [ratios[m] for m in middle], scale.rel_sigma)
        details = {**details, "used_flights": len(fits), "flight_errors": errors, "g_ratio": ratios}
        return unscored(type(self), inputs, SUBJECT, magnitude, sigma, span, details)

    def _flight_g(self, states: ObjectStateTable, flight: Flight) -> BallisticFallFit | None:
        """The flight's ballistic GLS; with an unknown sigma, unweighted and with g_var NaN."""
        sigma = floored_sigma(states, flight, self.params.exact_sigma_lengths)
        rows = flight.rows
        if sigma is None:
            cov = np.broadcast_to(np.eye(3), (rows.size, 3, 3))
        else:
            cov = window_covariance(states, flight, sigma, self.params.exact_sigma_lengths)
        g_dir = np.asarray(states.gravity_dir, dtype=np.float64)
        fit = gls_ballistic_fit(states.t[rows], states.position[rows, flight.index], cov, g_dir)
        if fit is None or sigma is not None:
            return fit
        return BallisticFallFit(fit.g, math.nan)


PLUGINS = (LineOfSightGravityValueCheck,)
