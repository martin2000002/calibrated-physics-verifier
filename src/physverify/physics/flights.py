"""Free flights of a track: rows between contacts and rest, cut at position jumps, and the
parabola fitted to each."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from physverify.physics.finite_diff import second_difference
from physverify.physics.kinematics import visible_runs
from physverify.physics.measures.gravity_fit import FallFit, fall_fit
from physverify.physics.tracks import (
    MIN_FIT_ROWS,
    IntArray,
    flight_mask,
    native_span,
    rest_mask,
    segments,
)
from physverify.schema.contacts import ContactTable
from physverify.schema.states import ObjectStateTable


def split_at_jumps(
    states: ObjectStateTable, index: int, seg: tuple[int, int], gate: float
) -> list[tuple[int, int]]:
    """Cut the row segment where ||a_k|| > gate * median ||a||.

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
