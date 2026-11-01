"""R5a whitened: v1's candidates, each scored by the Wald contrast of the two accelerations fitted
with weights from the per-row position sigma instead of their spread over |a|."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks.r5a_fall_shape import Candidate, FallShapeCheck, FallShapeParams
from physverify.physics.flights import Flight
from physverify.physics.measures.weighted_fall import (
    WeightedFallFit,
    accel_wald,
    wald_terms,
    weighted_fall_fit,
)
from physverify.physics.tracks import body_length
from physverify.schema.checks import AbstainReason
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]

EXACT_SIGMA_LENGTHS = 1e-6
"""Body lengths: the sigma an exact (zero-sigma) position is weighted with, on the scale of
``min_drop_lengths``. Oracle and declared-exact tables then keep finite weights; it is fixed a
priori, not measured. A perceived position always carries sigma > 0, so the floor only serves
oracle inputs. In a window that mixes both, the exact rows weigh (sigma / floor)^2 more, fix the
fit and its variance alone: W then measures the contrast against the floor and reads any noise
on those rows as a change of gravity."""


class WhitenedFallShapeParams(FallShapeParams):
    exact_sigma_lengths: float = Field(default=EXACT_SIGMA_LENGTHS, gt=0.0)


class WhitenedFallShapeCheck(FallShapeCheck):
    """R5a whitened [accel_wald_chi2_3]: per candidate of v1 (split, consecutive flights, pair of
    first flights), each window gets a parabola per axis weighted by 1 / sigma^2 of its rows and
    the magnitude is W = sum_d (a1_d - a2_d)^2 / (var1_d + var2_d), chi-square with three degrees
    of freedom when both windows share one acceleration and sigma is the position noise. W
    compares the contrast with what the window's own noise allows, so a clip with a noisier depth
    no longer reads as a larger change of gravity. A global scale of sigma rescales W without
    changing its order.

    The perturbations it answers to are injected in ``accel_rel_diff``: each result carries v1's
    ratio of the chosen windows as that estimate. A row with a finite position and an unknown
    sigma leaves its window without a statistic; an object left with none abstains with
    UNKNOWN_UNCERTAINTY."""

    variant: ClassVar[str] = "whitened"
    magnitude_unit: ClassVar[str] = "accel_wald_chi2_3"
    perturbation_unit: ClassVar[str] = "accel_rel_diff"
    perturbation_estimate: ClassVar[str] = "accel_rel_diff_estimated"
    Params: ClassVar[type[BaseModel]] = WhitenedFallShapeParams
    params: WhitenedFallShapeParams

    def __init__(self, params: BaseModel) -> None:
        self.params = WhitenedFallShapeParams.model_validate(params)

    def _statistic(
        self, states: ObjectStateTable, windows: Sequence[Flight], offsets: Sequence[Any]
    ) -> float:
        first, second = (self._fit(states, w, o) for w, o in zip(windows, offsets, strict=True))
        if first is None or second is None:
            return math.nan
        return accel_wald(first, second)

    def _no_candidate(self, states: ObjectStateTable, flights: Sequence[Flight]) -> AbstainReason:
        if any(self._window_sigma(states, f) is None for f in flights):
            return AbstainReason.UNKNOWN_UNCERTAINTY
        return super()._no_candidate(states, flights)

    def _statistic_details(self, states: ObjectStateTable, best: Candidate) -> dict[str, Any]:
        estimate = {self.perturbation_estimate: super()._statistic(states, best.windows, (0, 0))}
        first, second = (self._fit(states, w, 0.0) for w in best.windows)
        if first is None or second is None:
            return estimate
        return estimate | {
            "wald_per_axis": wald_terms(first, second),
            "weighted_accelerations": [first.a, second.a],
            "accel_var": [first.a_var, second.a_var],
        }

    def _fit(self, states: ObjectStateTable, window: Flight, offset: Any) -> WeightedFallFit | None:
        sigma = self._window_sigma(states, window)
        if sigma is None:
            return None
        positions = states.position[window.rows, window.index] + offset
        return weighted_fall_fit(states.t[window.rows], positions, sigma)

    def _window_sigma(self, states: ObjectStateTable, window: Flight) -> FloatArray | None:
        """The window's position sigma, exact rows floored at ``exact_sigma_lengths`` body
        lengths; None when a row with a finite position has no valid sigma."""
        sigma = np.asarray(states.position_sigma[window.rows, window.index], dtype=np.float64)
        seen = np.isfinite(states.position[window.rows, window.index])
        if bool(np.any(seen & ~(np.isfinite(sigma) & (sigma >= 0.0)))):
            return None
        exact = sigma == 0.0
        if not exact.any():
            return sigma
        length = body_length(states, window.index)
        floor = self.params.exact_sigma_lengths * (length if math.isfinite(length) else 1.0)
        return np.where(exact, floor, sigma)


PLUGINS = (WhitenedFallShapeCheck,)
