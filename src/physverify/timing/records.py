"""Records of a timing run: samples, failures, per-target outcomes and ``results.json``.

A measure is named ``<target prefix>[.<part>]*.<leaf>`` and its suffix carries the unit:
``_s`` seconds, ``_mb`` megabytes (MiB).
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from physverify.core.errors import TimingError
from physverify.timing.settings import TargetRunSettings

SCHEMA_VERSION: Final = "physverify.timing/1"
MIB: Final = float(1 << 20)
UNIT_BY_SUFFIX: Final[Mapping[str, str]] = MappingProxyType({"_s": "s", "_mb": "MB"})
MEASURE_PATTERN: Final = r"^[a-z0-9_]+(\.[a-z0-9_]+)*_(s|mb)$"
_MEASURE_RE: Final = re.compile(MEASURE_PATTERN)

# Measures written by one package and read by another (harness, targets, report, budget).
PREPARE_S: Final = "setup.prepare_s"
GENERATE_LOAD_S: Final = "generate.load_s"
GENERATE_LOAD_CUDA_MB: Final = "generate.load_cuda_mb"
EXTRACT_WALL_S: Final = "extract.wall_s"
EXTRACT_SPAN_S: Final = "extract.span_s"
EXTRACT_COLD_EXTRA_S: Final = "extract.cold_extra_s"
EXTRACT_PEAK_RSS_MB: Final = "extract.peak_rss_mb"
VERIFY_WALL_S: Final = "verify.wall_s"
VERIFY_LOAD_S: Final = "verify.load_s"
VERIFY_CHECKS_S: Final = "verify.checks_s"
VERIFY_CPU_S: Final = "verify.cpu_s"
RENDER_WALL_S: Final = "render.wall_s"
RENDER_PER_FRAME_S: Final = "render.per_frame_s"


def unit_of(measure: str) -> str:
    if _MEASURE_RE.match(measure) is None:
        raise TimingError(f"invalid measure name {measure!r}: expected {MEASURE_PATTERN}")
    return next(unit for suffix, unit in UNIT_BY_SUFFIX.items() if measure.endswith(suffix))


def measure_name(*parts: str) -> str:
    """``measure_name("extract", "depth", "wall_s")`` -> ``"extract.depth.wall_s"``, validated."""
    name = ".".join(parts)
    unit_of(name)
    return name


def generate_measure(batch_size: int, leaf: str) -> str:
    return measure_name("generate", f"b{batch_size}", leaf)


def check_measures(values: Mapping[str, float]) -> None:
    for name, value in values.items():
        unit_of(name)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TimingError(f"{name}: expected a number, got {type(value).__name__}")
        if not math.isfinite(float(value)) or value < 0:
            raise TimingError(f"{name} = {value!r}: expected a finite value >= 0")


@dataclass(frozen=True)
class Sample:
    target: str
    clip_id: str
    repetition: int
    warmup: bool
    measure: str
    unit: str
    value: float


@dataclass(frozen=True)
class CallFailure:
    target: str
    clip_id: str
    repetition: int
    error: str


class TargetStatus(StrEnum):
    MEASURED = "measured"
    SKIPPED = "skipped"
    FAILED = "failed"
    PARTIAL = "partial"  # some calls failed, but at least one clip has samples


@dataclass(frozen=True)
class TargetOutcome:
    name: str
    requires_gpu: bool
    status: TargetStatus
    reason: str | None
    run: TargetRunSettings
    clip_ids: tuple[str, ...] = ()
    info: Mapping[str, str] = MappingProxyType({})
    setup: Mapping[str, float] = MappingProxyType({})  # once per run, e.g. generate.load_s
    failures: tuple[CallFailure, ...] = ()
    n_calls: int = 0


class _Record(BaseModel):
    # sd is NaN with one sample and iqr_rel with a zero median: keep them as NaN in JSON.
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="constants")


class MeasureStats(_Record):
    """One statistic row (a clip, or ``clip_id="*"`` across clips); the columns of stats.csv."""

    target: str
    clip_id: str
    measure: str
    unit: str
    n: int = Field(ge=1)
    median: float
    p25: float
    p75: float
    iqr: float
    iqr_rel: float
    mean: float
    sd: float
    min: float
    max: float
    unstable: bool


class TargetResult(_Record):
    name: str
    requires_gpu: bool
    status: TargetStatus
    reason: str | None = None
    run: TargetRunSettings
    clip_ids: list[str] = Field(default_factory=list)
    info: dict[str, str] = Field(default_factory=dict)
    setup: dict[str, float] = Field(default_factory=dict)
    failures: list[CallFailure] = Field(default_factory=list)
    n_calls: int = Field(0, ge=0)
    stats: list[MeasureStats] = Field(default_factory=list)

    @classmethod
    def from_outcome(
        cls, outcome: TargetOutcome, stats: Sequence[MeasureStats] = ()
    ) -> TargetResult:
        return cls(
            name=outcome.name,
            requires_gpu=outcome.requires_gpu,
            status=outcome.status,
            reason=outcome.reason,
            run=outcome.run,
            clip_ids=list(outcome.clip_ids),
            info=dict(outcome.info),
            setup=dict(outcome.setup),
            failures=list(outcome.failures),
            n_calls=outcome.n_calls,
            stats=list(stats),
        )

    def outcome(self) -> TargetOutcome:
        return TargetOutcome(
            name=self.name,
            requires_gpu=self.requires_gpu,
            status=self.status,
            reason=self.reason,
            run=self.run,
            clip_ids=tuple(self.clip_ids),
            info=MappingProxyType(dict(self.info)),
            setup=MappingProxyType(dict(self.setup)),
            failures=tuple(self.failures),
            n_calls=self.n_calls,
        )


class ResultsFile(_Record):
    """Shape of ``results.json``; ``targets`` grows as the run goes (empty at start)."""

    schema_version: Literal["physverify.timing/1"] = SCHEMA_VERSION
    valid_measurement: bool
    validity_reasons: list[str]
    run: dict[str, Any]
    environment: dict[str, Any]
    targets: list[TargetResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> ResultsFile:
        if self.valid_measurement == bool(self.validity_reasons):
            raise ValueError(
                "valid_measurement must be true exactly when validity_reasons is empty"
            )
        return self
