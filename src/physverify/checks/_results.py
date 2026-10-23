"""Result builders for the checks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from physverify.checks._base import Check, CheckInputs
from physverify.checks._params import TIE_RTOL
from physverify.physics.tracks import row_step
from physverify.schema.checks import AbstainReason, CheckResult, Verdict

FloatArray = NDArray[np.float64]


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
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("first_max needs at least one finite value")
    peak = float(finite.max())
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
