"""Result of one physical check on one subject of one clip."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from physverify.schema._validate import (
    check_enum,
    check_json_mapping,
    check_number,
    check_sigma,
    check_span,
    check_str,
    check_str_tuple,
)
from physverify.schema.errors import SchemaError
from physverify.schema.io import Record, optional

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"
    UNSCORED = "UNSCORED"


class AbstainReason(str, Enum):
    MISSING_INPUT = "MISSING_INPUT"
    SCALE_FROM_GRAVITY = "SCALE_FROM_GRAVITY"
    SCALE_UNAVAILABLE = "SCALE_UNAVAILABLE"
    TOO_FEW_FRAMES = "TOO_FEW_FRAMES"
    ILL_CONDITIONED = "ILL_CONDITIONED"
    OCCLUDED = "OCCLUDED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    CONTACT_GEOMETRY_UNKNOWN = "CONTACT_GEOMETRY_UNKNOWN"
    UNKNOWN_UNCERTAINTY = "UNKNOWN_UNCERTAINTY"
    INTERVAL_CROSSES_THRESHOLD = "INTERVAL_CROSSES_THRESHOLD"


class InputLevel(str, Enum):
    ORACLE = "oracle"
    PERCEIVED = "perceived"
    MIXED = "mixed"


@dataclass(frozen=True, eq=False)
class CheckResult(Record):
    """Checks emit UNSCORED (a magnitude) or ABSTAIN; a VerdictPolicy sets PASS/FAIL.

    ``subject`` is an object id, an object pair or empty (whole clip). ``provenance`` maps each
    stage to its provider plus ``scale.source``.
    """

    check_id: str
    clip_id: str
    subject: tuple[str, ...]
    frame_span: tuple[int, int] | None
    magnitude: float | None
    sigma: float
    unit: str
    verdict: Verdict
    input_level: InputLevel
    abstain_reason: AbstainReason | None = None
    threshold: float | None = None
    provenance: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        check_str("check_id", self.check_id)
        check_str("clip_id", self.clip_id)
        check_str_tuple("subject", self.subject)
        if self.frame_span is not None:
            check_span("frame_span", self.frame_span)
        if self.magnitude is not None:
            check_number("magnitude", self.magnitude)
        check_sigma("sigma", self.sigma)
        check_str("unit", self.unit, allow_empty=True)
        check_enum("verdict", self.verdict, Verdict)
        check_enum("input_level", self.input_level, InputLevel)
        self._validate_abstention()
        if self.threshold is not None:
            check_number("threshold", self.threshold)
        check_json_mapping("provenance", self.provenance)
        check_json_mapping("details", self.details)

    def _validate_abstention(self) -> None:
        abstained = self.verdict is Verdict.ABSTAIN
        if abstained != (self.abstain_reason is not None):
            raise SchemaError("abstain_reason is required exactly when verdict is ABSTAIN")
        if self.abstain_reason is not None:
            check_enum("abstain_reason", self.abstain_reason, AbstainReason)
        if not abstained and self.magnitude is None:
            raise SchemaError("a non-abstaining result needs a magnitude")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            check_id=data["check_id"],
            clip_id=data["clip_id"],
            subject=tuple(data["subject"]),
            frame_span=optional(data["frame_span"], tuple),
            magnitude=data["magnitude"],
            sigma=data["sigma"],
            unit=data["unit"],
            verdict=Verdict(data["verdict"]),
            input_level=InputLevel(data["input_level"]),
            abstain_reason=optional(data["abstain_reason"], AbstainReason),
            threshold=data["threshold"],
            provenance=data["provenance"],
            details=data["details"],
        )
