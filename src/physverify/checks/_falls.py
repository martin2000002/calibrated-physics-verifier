"""Fall search shared by R5a and R5b: flights that pass the travel filter."""

from __future__ import annotations

import math

import numpy as np

from physverify.checks._params import FallParams
from physverify.physics.flights import Flight, find_flights
from physverify.physics.tracks import FloatArray, body_length, track_noise
from physverify.schema.contacts import ContactTable
from physverify.schema.states import ObjectStateTable


def find_falls(
    states: ObjectStateTable, contacts: ContactTable, index: int, params: FallParams
) -> list[Flight]:
    """Flights of object ``index`` that pass the fall filter of ``params``, in time order."""
    flights = find_flights(states, contacts, index, params.rest_speed, params.jump_gate)
    noise = (
        track_noise(states, contacts, index, params.min_noise_stencils)
        if params.travel_noise == "track"
        else None
    )
    return [f for f in flights if _travel(states, f) > _travel_floor(states, f, params, noise)]


def _travel(states: ObjectStateTable, flight: Flight) -> float:
    positions = states.position[flight.rows, flight.index]
    return float(np.max(np.linalg.norm(positions - positions[0], axis=-1)))


def _travel_floor(
    states: ObjectStateTable, flight: Flight, params: FallParams, track: FloatArray | None
) -> float:
    """``track``: the object's per-axis noise (None outside "track" mode or without stencils)."""
    length = body_length(states, flight.index)
    by_length = params.min_travel_lengths * length if math.isfinite(length) else 0.0
    sigma = states.position_sigma[flight.rows, flight.index]
    known = sigma[np.all(np.isfinite(sigma), axis=-1)]
    exact = known.shape[0] > 0 and bool(np.all(known == 0.0))
    if params.travel_noise == "track" and track is not None and not exact:
        return max(by_length, params.travel_sigmas * math.sqrt(2.0) * float(np.linalg.norm(track)))
    if not known.shape[0]:
        return by_length
    noise = math.sqrt(2.0) * float(np.linalg.norm(np.median(known, axis=0)))
    return max(by_length, params.travel_sigmas * noise)
