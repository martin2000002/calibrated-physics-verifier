"""Result builders, per-check randomness and Monte Carlo sigma for the checks (ADR 0006 3.3)."""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from physverify.checks._base import Check, CheckInputs
from physverify.config.models import FrozenModel
from physverify.physics.tracks import Flight, body_length, find_flights, row_step
from physverify.physics.uncertainty import monte_carlo
from physverify.schema.checks import AbstainReason, CheckResult, Verdict
from physverify.schema.contacts import ContactTable
from physverify.schema.states import ObjectStateTable

FloatArray = NDArray[np.float64]
# Relative slack under which two per-row values count as the same maximum, so ties (a teleport
# inflates two consecutive rows equally) resolve to the earliest row at any row step.
TIE_RTOL = 1e-9


class CommonParams(FrozenModel):
    """Fields every check's ``Params`` inherits."""

    mc_draws: int = Field(default=200, ge=2)
    seed: int = Field(default=0, ge=0)


class OcclusionParams(CommonParams):
    """Occlusion fraction from which an object counts as hidden (checks that read occlusion)."""

    occluded_at: float = Field(default=0.9, gt=0.0, le=1.0)


class FallParams(CommonParams):
    """Flight search and fall filter shared by R5a and R5b.

    A flight is a fall when the observed travel max_k |p_k - p_first| exceeds both
    ``min_travel_lengths`` body lengths and ``travel_sigmas`` times the noise of a position
    difference (sqrt(2) |median position sigma|). The fitted acceleration is deliberately not
    used: R5b tests that acceleration, and filtering on it would drop weak-gravity flights.
    """

    rest_speed: float = Field(default=0.1, ge=0.0)
    jump_gate: float = Field(default=3.0, gt=1.0)
    min_travel_lengths: float = Field(default=0.5, ge=0.0)
    travel_sigmas: float = Field(default=3.0, ge=0.0)


def find_falls(
    states: ObjectStateTable, contacts: ContactTable, index: int, params: FallParams
) -> list[Flight]:
    """Flights of object ``index`` that pass the fall filter of ``params``, in time order."""
    flights = find_flights(states, contacts, index, params.rest_speed, params.jump_gate)
    return [f for f in flights if _travel(states, f) > _travel_floor(states, f, params)]


def _travel(states: ObjectStateTable, flight: Flight) -> float:
    positions = states.position[flight.rows, flight.index]
    return float(np.max(np.linalg.norm(positions - positions[0], axis=-1)))


def _travel_floor(states: ObjectStateTable, flight: Flight, params: FallParams) -> float:
    length = body_length(states, flight.index)
    by_length = params.min_travel_lengths * length if math.isfinite(length) else 0.0
    sigma = states.position_sigma[flight.rows, flight.index]
    known = np.all(np.isfinite(sigma), axis=-1)
    if not bool(known.any()):
        return by_length
    noise = math.sqrt(2.0) * float(np.linalg.norm(np.median(sigma[known], axis=0)))
    return max(by_length, params.travel_sigmas * noise)


def unscored(
    check: type[Check],
    inputs: CheckInputs,
    subject: tuple[str, ...],
    magnitude: float,
    sigma: float,
    frame_span: tuple[int, int] | None,
    details: Mapping[str, Any],
) -> CheckResult:
    return _result(
        check,
        inputs,
        subject,
        frame_span,
        details,
        magnitude=float(magnitude),
        sigma=float(sigma),
        verdict=Verdict.UNSCORED,
        abstain_reason=None,
    )


def abstain(
    check: type[Check],
    inputs: CheckInputs,
    subject: tuple[str, ...],
    reason: AbstainReason,
    frame_span: tuple[int, int] | None,
    details: Mapping[str, Any],
) -> CheckResult:
    return _result(
        check,
        inputs,
        subject,
        frame_span,
        details,
        magnitude=None,
        sigma=math.nan,
        verdict=Verdict.ABSTAIN,
        abstain_reason=reason,
    )


def _result(
    check: type[Check],
    inputs: CheckInputs,
    subject: tuple[str, ...],
    frame_span: tuple[int, int] | None,
    details: Mapping[str, Any],
    *,
    magnitude: float | None,
    sigma: float,
    verdict: Verdict,
    abstain_reason: AbstainReason | None,
) -> CheckResult:
    return CheckResult(
        check_id=check.check_id,
        clip_id=inputs.clip.clip_id,
        subject=tuple(subject),
        frame_span=None if frame_span is None else (int(frame_span[0]), int(frame_span[1])),
        magnitude=magnitude,
        sigma=sigma,
        unit=check.magnitude_unit,
        verdict=verdict,
        input_level=inputs.input_level,
        abstain_reason=abstain_reason,
        provenance=dict(inputs.provenance),
        details={**_json_native(details), "row_step": row_step_or_none(inputs)},
    )


def row_step_or_none(inputs: CheckInputs) -> int | None:
    """``row_step`` of the table; None when its frame_index is not uniform."""
    try:
        return row_step(inputs.states.frame_index)
    except ValueError:
        return None


def first_max(values: FloatArray) -> int:
    """Index of the first value within TIE_RTOL of the largest finite one."""
    peak = float(np.nanmax(values))
    return int(np.flatnonzero(values >= peak - TIE_RTOL * abs(peak))[0])


def _json_native(value: Any) -> Any:
    """NumPy scalars and arrays, tuples and mappings as plain JSON values."""
    if isinstance(value, Mapping):
        return {str(k): _json_native(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def check_rng(seed: int, clip_id: str, check_id: str) -> np.random.Generator:
    """Independent, reproducible stream per (seed, clip, check)."""
    return np.random.default_rng(
        [seed, zlib.crc32(clip_id.encode()), zlib.crc32(check_id.encode())]
    )


def mc_sigma(
    model: Callable[[Mapping[str, FloatArray]], FloatArray],
    sigmas: Mapping[str, FloatArray],
    draws: int,
    rng: np.random.Generator,
) -> float:
    """Std of ``model`` under N(0, sigma) noise on each input.

    ``model`` receives one (draws, *shape) noise array per key and returns (draws,) magnitudes.
    Exact inputs give 0; any NaN sigma makes the result NaN.
    """
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in sigmas.items()}
    if all(bool(np.all(value == 0.0)) for value in arrays.values()):
        return 0.0
    noise = {key: _noise(value, draws, rng) for key, value in arrays.items()}
    if all(value is None for value in noise.values()):
        return math.nan
    std = np.asarray(monte_carlo(model, noise).std, dtype=np.float64)
    if std.size != 1:
        raise ValueError(f"model must return one magnitude per draw, got std shape {std.shape}")
    return float(std.reshape(()))


def _noise(sigma: FloatArray, draws: int, rng: np.random.Generator) -> FloatArray | None:
    if not bool(np.all(np.isfinite(sigma))):
        return None
    return rng.normal(0.0, 1.0, size=(draws, *sigma.shape)) * sigma
