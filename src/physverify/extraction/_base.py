"""Stage execution contract: providers, their context, weights and observers (ADR 0003, 3.2).

Heavy libraries (torch, transformers) are imported inside ``run()``, never at module level.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from physverify.config.models import PathsSettings
from physverify.core.errors import InvalidPluginError, MissingInputError
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.schema.clip import ClipManifest, FrameSampling
from physverify.schema.error_profile import ErrorEntry, ErrorProfile
from physverify.schema.errors import SchemaError
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import Stage, check_upstream_order, stage_index

GROUND_TRUTH = "ground_truth"
"""Provider name of every conditional ground-truth provider."""

O = TypeVar("O")  # noqa: E741


class WeightKind(StrEnum):
    HF = "hf"
    URL = "url"
    GIT = "git"


@dataclass(frozen=True)
class WeightSpec:
    """A checkpoint or repository a provider downloads with ``extract fetch-weights``."""

    kind: WeightKind
    ref: str
    revision: str = ""
    sha256: str = ""


@dataclass(frozen=True)
class StageContext:
    """Everything a provider may read; ``upstream`` holds only the stages it requires."""

    clip: ClipManifest
    video: VideoSource
    sampling: FrameSampling
    upstream: Mapping[Stage, StageOutput]
    plan: Mapping[Stage, str]
    gt: GroundTruthStore | None
    error_profile: ErrorProfile | None
    device: str
    threads: int
    paths: PathsSettings

    def require(self, stage: Stage, kind: type[O]) -> O:
        output = self.upstream.get(stage)
        if output is None:
            raise MissingInputError(f"{self.clip.clip_id}: upstream {stage.value} is missing")
        if not isinstance(output, kind):
            raise MissingInputError(
                f"{self.clip.clip_id}: upstream {stage.value} is {type(output).__name__}, "
                f"expected {kind.__name__}"
            )
        return output

    def require_gt(self) -> GroundTruthStore:
        if self.gt is None:
            raise MissingInputError(f"{self.clip.clip_id}: no ground truth for this clip")
        return self.gt

    def error_entry(self, stage: Stage, quantity: str) -> ErrorEntry | None:
        """Error of the provider that ran ``stage`` in the active plan (None = unknown sigma).

        A ground-truth upstream contributes zero error."""
        provider = self.plan.get(stage)
        if provider is None:
            return None
        if provider == GROUND_TRUTH:
            return ErrorEntry.exact(stage, GROUND_TRUTH, quantity)
        if self.error_profile is None:
            return None
        return self.error_profile.lookup(stage, provider, quantity)


def error_profile_keys(
    reads: tuple[Stage, ...], plan: Mapping[Stage, str]
) -> tuple[tuple[Stage, str], ...]:
    """(stage, provider) profile keys a provider declares; ground truth has no entry."""
    stages = sorted((s for s in reads if s in plan and plan[s] != GROUND_TRUTH), key=stage_index)
    return tuple((stage, plan[stage]) for stage in stages)


@runtime_checkable
class StageProvider(Protocol):
    """A perception model or conditional ground truth for one stage; key ``<stage>/<name>``."""

    stage: ClassVar[Stage]
    name: ClassVar[str]
    version: ClassVar[str]
    requires: ClassVar[tuple[Stage, ...]]
    heavy: ClassVar[bool]
    is_ground_truth: ClassVar[bool]
    error_profile_reads: ClassVar[tuple[Stage, ...]]
    weights: ClassVar[tuple[WeightSpec, ...]]
    Params: ClassVar[type[BaseModel]]

    def __init__(self, params: BaseModel) -> None: ...

    def run(self, ctx: StageContext) -> StageOutput: ...


def provider_key(provider: type[StageProvider]) -> str:
    return f"{provider.stage.value}/{provider.name}"


def validate_provider(provider: type[StageProvider]) -> None:
    """Registry hook: dependency order, ground-truth naming and parameter model."""
    try:
        check_upstream_order(provider.stage, provider.requires)
        check_upstream_order(provider.stage, provider.error_profile_reads)
    except SchemaError as exc:
        raise InvalidPluginError(f"{provider.__name__}: {exc}") from exc
    if provider.is_ground_truth != (provider.name == GROUND_TRUTH):
        raise InvalidPluginError(f"{provider.__name__}: ground truth iff name == {GROUND_TRUTH!r}")
    if not (isinstance(provider.Params, type) and issubclass(provider.Params, BaseModel)):
        raise InvalidPluginError(f"{provider.__name__}: Params must be a pydantic model")


@dataclass(frozen=True)
class StageEvent:
    clip_id: str
    stage: Stage
    provider: str
    cached: bool
    device: str
    wall_s: float = math.nan
    peak_rss_mb: float = math.nan


@runtime_checkable
class StageObserver(Protocol):
    """Receives stage timings (used by the timing harness)."""

    def on_stage_start(self, event: StageEvent) -> None: ...

    def on_stage_end(self, event: StageEvent) -> None: ...
