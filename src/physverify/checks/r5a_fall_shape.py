"""R5a scale-free fall shape: the parabola fitted on a flight has one constant acceleration,
shared by consecutive flights and by the objects of the clip. ADR 0006 section 3.4.

Flights are the rows outside every contact window of the object and not at rest, cut where a
row's acceleration jumps (D12). Every fit uses the table time ``states.t``, so strided tables
give the same accelerations as native ones (D14.2).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs
from physverify.checks._falls import find_falls
from physverify.checks._params import FallParams
from physverify.checks._results import abstain, first_max, unscored
from physverify.checks._uncertainty import check_rng, mc_sigma
from physverify.physics.measures.gravity_fit import (
    accel_inconsistency,
    fall_residual_ratio,
    lateral_accel_ratio,
)
from physverify.physics.tracks import (
    MIN_FIT_ROWS,
    Flight,
    body_length,
    fit_flight,
    flights_span,
    native_span,
)
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.contacts import ContactTable
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class FallShapeParams(FallParams):
    min_flight_frames: int = Field(default=6, ge=MIN_FIT_ROWS)
    side_frames: int = Field(default=4, ge=MIN_FIT_ROWS)
    # After the shared fall filter, a flight whose fitted drop |a| T^2 / 2 stays below this many
    # body lengths has no acceleration to shape (a drift without gravity): its ratio is 0 / 0.
    min_drop_lengths: float = Field(default=1e-6, ge=0.0)


@dataclass(frozen=True, eq=False)
class Candidate:
    """Inconsistency between the accelerations fitted on two row windows."""

    value: float
    windows: tuple[Flight, Flight]
    flights: tuple[Flight, ...]
    mode: str
    split_row: int | None = None


class FallShapeCheck:
    """R5a [accel_rel_diff]: per object, the largest inconsistency between the two sides of a
    split flight or between consecutive flights; per object pair, between their first flights.
    Does not need the gravity direction (D10): ``lateral_accel_ratio`` is null without it.
    Under noise the maximum over candidates is biased upward and its sigma is the chosen
    candidate's own, not the sigma of the maximum (a known limitation, left to calibration)."""

    check_id: ClassVar[str] = "R5a"
    requires: ClassVar[frozenset[str]] = frozenset({"contacts"})
    magnitude_unit: ClassVar[str] = "accel_rel_diff"
    Params: ClassVar[type[BaseModel]] = FallShapeParams

    def __init__(self, params: BaseModel) -> None:
        self.params = FallShapeParams.model_validate(params)

    def evaluate(self, inputs: CheckInputs) -> list[CheckResult]:
        if inputs.contacts is None:
            raise ValueError(f"{self.check_id} requires contacts")
        states, contacts = inputs.states, inputs.contacts
        rng = check_rng(self.params.seed, inputs.clip.clip_id, self.check_id)
        falls = {i: self._falls(states, contacts, i) for i in range(states.n_objects)}
        results = [self._object(inputs, flights, rng) for flights in falls.values() if flights]
        firsts = [long[0] for long in (self._long(f) for f in falls.values()) if long]
        for a, b in itertools.combinations(firsts, 2):
            pair = _candidate(states, (a, b), (a, b), "pair")
            if math.isfinite(pair.value):
                results.append(self._scored(inputs, _ids(states, a, b), pair, rng))
        return results

    def _falls(self, states: ObjectStateTable, contacts: ContactTable, index: int) -> list[Flight]:
        flights = find_falls(states, contacts, index, self.params)
        return [f for f in flights if self._accelerates(states, f)]

    def _accelerates(self, states: ObjectStateTable, flight: Flight) -> bool:
        fit = fit_flight(states, flight)
        duration = float(states.t[flight.rows[-1]] - states.t[flight.rows[0]])
        drop = 0.5 * float(np.linalg.norm(fit.a)) * duration**2
        length = body_length(states, flight.index)
        floor = self.params.min_drop_lengths * length if math.isfinite(length) else 0.0
        return drop > floor

    def _long(self, flights: Sequence[Flight]) -> list[Flight]:
        return [f for f in flights if f.present >= self.params.min_flight_frames]

    def _object(
        self, inputs: CheckInputs, flights: Sequence[Flight], rng: np.random.Generator
    ) -> CheckResult:
        states = inputs.states
        long = self._long(flights)
        candidates = [c for f in long for c in self._splits(states, f)]
        candidates += [
            _candidate(states, (a, b), (a, b), "consecutive") for a, b in itertools.pairwise(long)
        ]
        candidates = [c for c in candidates if math.isfinite(c.value)]
        subject = (str(states.object_ids[flights[0].index]),)
        if not candidates:
            details = {
                "flights": len(flights),
                "longest_rows": max(f.present for f in flights),
                "min_flight_frames": self.params.min_flight_frames,
                "side_frames": self.params.side_frames,
            }
            span = flights_span(states.frame_index, flights)
            reason = AbstainReason.TOO_FEW_FRAMES
            return abstain(type(self), inputs, subject, reason, span, details)
        best = candidates[first_max(np.array([c.value for c in candidates]))]
        return self._scored(inputs, subject, best, rng)

    def _splits(self, states: ObjectStateTable, flight: Flight) -> list[Candidate]:
        """One candidate per row k leaving >= side_frames present rows on each side."""
        side = self.params.side_frames
        out = []
        for k in range(flight.lo + 1, flight.hi):
            left, right = flight.rows[flight.rows < k], flight.rows[flight.rows >= k]
            if left.size >= side and right.size >= side:
                windows = (_sub(flight, left, k), _sub(flight, right, flight.hi))
                out.append(_candidate(states, windows, (flight,), "split", split_row=k))
        return out

    def _scored(
        self,
        inputs: CheckInputs,
        subject: tuple[str, ...],
        best: Candidate,
        rng: np.random.Generator,
    ) -> CheckResult:
        states = inputs.states
        sigma = _sigma(states, best.windows, self.params.mc_draws, rng)
        span = flights_span(states.frame_index, best.flights)
        details = _details(states, best)
        return unscored(type(self), inputs, subject, best.value, sigma, span, details)


def _sub(flight: Flight, rows: IntArray, hi: int) -> Flight:
    return Flight(flight.index, int(rows[0]), hi, rows)


def _ids(states: ObjectStateTable, a: Flight, b: Flight) -> tuple[str, str]:
    return str(states.object_ids[a.index]), str(states.object_ids[b.index])


def _candidate(
    states: ObjectStateTable,
    windows: tuple[Flight, Flight],
    flights: tuple[Flight, ...],
    mode: str,
    split_row: int | None = None,
) -> Candidate:
    value = _inconsistency(states, windows, (0.0, 0.0))
    return Candidate(value, windows, flights, mode, split_row)


def _inconsistency(
    states: ObjectStateTable, windows: Sequence[Flight], offsets: Sequence[Any]
) -> float:
    fits = [fit_flight(states, w, o) for w, o in zip(windows, offsets, strict=True)]
    try:
        return accel_inconsistency([fit.a for fit in fits])
    except ZeroDivisionError:
        return math.nan


def _sigma(
    states: ObjectStateTable,
    windows: Sequence[Flight],
    draws: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo over independent position noise on the rows of each fitted window."""
    sigmas = {str(n): states.position_sigma[w.rows, w.index] for n, w in enumerate(windows)}

    def model(noise: Mapping[str, FloatArray]) -> FloatArray:
        per_window = [noise[str(n)] for n in range(len(windows))]
        return np.array(
            [_inconsistency(states, windows, [x[d] for x in per_window]) for d in range(draws)]
        )

    return mc_sigma(model, sigmas, draws, rng)


def _details(states: ObjectStateTable, best: Candidate) -> dict[str, Any]:
    fi = states.frame_index
    fits = [fit_flight(states, f) for f in best.flights]
    residuals = [
        fall_residual_ratio(fit, float(states.t[f.hi - 1] - states.t[f.lo]))
        for f, fit in zip(best.flights, fits, strict=True)
    ]
    known_gravity = bool(np.all(np.isfinite(states.gravity_dir)))
    lateral = [lateral_accel_ratio(fit.a, states.gravity_dir) for fit in fits]
    return {
        "mode": best.mode,
        "split_frame": None if best.split_row is None else int(fi[best.split_row]),
        "windows": [native_span(fi, int(w.rows[0]), int(w.rows[-1])) for w in best.windows],
        "accelerations": [fit_flight(states, w).a for w in best.windows],
        "fall_residual_ratio": max(residuals),
        "lateral_accel_ratio": max(lateral) if known_gravity else None,
    }


PLUGINS = (FallShapeCheck,)
