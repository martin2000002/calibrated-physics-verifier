"""Physics check contract. A check returns magnitudes (UNSCORED) or ABSTAIN; the verdict is
set later by a ``VerdictPolicy``."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from physverify.config.plugin_context import PluginKind
from physverify.core.errors import InvalidPluginError
from physverify.physics.tracks import row_step
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest
from physverify.schema.contacts import ContactTable
from physverify.schema.perception import CameraModel
from physverify.schema.scale import EffectiveScale
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable


@dataclass(frozen=True)
class CheckInputs:
    """States and contacts in one unit and frame; ``scale`` converts that unit to meters;
    ``camera`` is the CAMERA output the states were lifted with (ground truth for the oracle)."""

    clip: ClipManifest
    states: ObjectStateTable
    contacts: ContactTable | None
    scale: EffectiveScale | None
    input_level: InputLevel
    provenance: Mapping[str, str]
    scene: SceneSpec | None = None
    camera: CameraModel | None = None


@runtime_checkable
class Check(Protocol):
    """One physics check (R1..R5b); registered by ``check_id``. ``time_dependent``: the magnitude
    depends on the clip's time base (fps)."""

    check_id: ClassVar[str]
    requires: ClassVar[frozenset[str]]
    time_dependent: ClassVar[bool]
    magnitude_unit: ClassVar[str]
    Params: ClassVar[type[BaseModel]]

    def __init__(self, params: BaseModel) -> None: ...

    def evaluate(self, inputs: CheckInputs) -> Sequence[CheckResult]: ...


def validate_check(plugin: type[Check]) -> None:
    """Registry hook: ``requires`` names fields of ``CheckInputs``."""
    unknown = sorted(set(plugin.requires) - {field.name for field in fields(CheckInputs)})
    if unknown:
        raise InvalidPluginError(f"{plugin.__name__}: requires unknown inputs {unknown}")


def missing_inputs(check: Check | type[Check], inputs: CheckInputs) -> list[str]:
    """The fields ``check`` requires that ``inputs`` lacks, sorted."""
    return sorted(name for name in check.requires if getattr(inputs, name) is None)


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


CHECKS = PluginKind[type[Check]]("checks")
