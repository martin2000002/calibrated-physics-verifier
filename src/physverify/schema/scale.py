"""Absolute scale with provenance and the single rule that applies it (effective_scale)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from physverify.schema._validate import (
    check_enum,
    check_number,
    check_positive,
    check_sigma,
    check_str,
)
from physverify.schema.errors import ScaleBindingError, SchemaError
from physverify.schema.io import Record

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from physverify.schema.states import ObjectStateTable


class ScaleSource(str, Enum):
    GROUND_TRUTH = "ground_truth"
    METRIC_DEPTH = "metric_depth"
    SIZE_PRIOR = "size_prior"
    FREE_FALL = "free_fall"
    NONE = "none"


@dataclass(frozen=True, eq=False)
class ScaleEstimate(Record):
    """Meters per table unit for the LIFT table whose ``source_key`` equals ``lift_key``.

    ``source=none`` has no scale (``meters_per_unit`` None). ``ground_truth`` has a finite
    sigma >= 0. Estimated sources may carry ``sigma=NaN`` until the error profile measures it.
    """

    meters_per_unit: float | None
    sigma: float
    source: ScaleSource
    lift_key: str
    detail: str = ""

    def validate(self) -> None:
        check_enum("source", self.source, ScaleSource)
        check_str("lift_key", self.lift_key)
        check_str("detail", self.detail, allow_empty=True)
        check_sigma("sigma", self.sigma)
        if self.source is ScaleSource.NONE:
            self._validate_none()
            return
        check_positive("meters_per_unit", self.meters_per_unit)
        if self.source is ScaleSource.GROUND_TRUTH and math.isnan(self.sigma):
            raise SchemaError("ground_truth scale needs a finite sigma >= 0")

    def _validate_none(self) -> None:
        if self.meters_per_unit is not None:
            raise SchemaError("source=none requires meters_per_unit None")
        if not math.isnan(check_number("sigma", self.sigma)):
            raise SchemaError("source=none requires sigma NaN")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            meters_per_unit=data["meters_per_unit"],
            sigma=data["sigma"],
            source=ScaleSource(data["source"]),
            lift_key=data["lift_key"],
            detail=data["detail"],
        )


@dataclass(frozen=True, eq=False)
class EffectiveScale(Record):
    """Scale ready to apply: meters = value x meters_per_unit; ``rel_sigma`` may be NaN."""

    meters_per_unit: float
    rel_sigma: float
    source: ScaleSource

    def validate(self) -> None:
        check_positive("meters_per_unit", self.meters_per_unit)
        check_sigma("rel_sigma", self.rel_sigma)
        check_enum("source", self.source, ScaleSource)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            meters_per_unit=data["meters_per_unit"],
            rel_sigma=data["rel_sigma"],
            source=ScaleSource(data["source"]),
        )


def effective_scale(states: ObjectStateTable, scale: ScaleEstimate | None) -> EffectiveScale | None:
    """None when there is no scale; ScaleBindingError when it was fitted on another table."""
    if scale is None or scale.source is ScaleSource.NONE:
        return None
    if scale.lift_key != states.source_key:
        raise ScaleBindingError(
            f"scale fitted on {scale.lift_key!r} applied to table {states.source_key!r}"
        )
    assert scale.meters_per_unit is not None
    return EffectiveScale(
        meters_per_unit=scale.meters_per_unit,
        rel_sigma=scale.sigma / scale.meters_per_unit,
        source=scale.source,
    )
