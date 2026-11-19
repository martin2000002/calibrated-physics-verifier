"""Timing target contract (wall time, CPU, peak memory of a workload)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from physverify.config.models import Settings
from physverify.config.plugin_context import PluginKind
from physverify.extraction._base import StageObserver
from physverify.schema.clip import ClipManifest

if TYPE_CHECKING:
    from physverify.timing.target_services import TimingServices


@dataclass(frozen=True)
class TimingContext:
    settings: Settings
    device: str
    threads: int
    run_dir: Path
    observer: StageObserver | None = None


@runtime_checkable
class TimingTarget(Protocol):
    """Registered by ``name``; built with the harness services; returns named measurements
    (seconds, MB). ``timing.hooks.TargetHooks`` gives the lifecycle methods no-op defaults."""

    name: ClassVar[str]
    requires_gpu: ClassVar[bool]

    def __init__(self, services: TimingServices) -> None: ...

    def run(self, clip: ClipManifest, ctx: TimingContext) -> Mapping[str, float]: ...

    def unavailable(self, ctx: TimingContext) -> str | None: ...

    def own_clips(self, ctx: TimingContext) -> Sequence[ClipManifest] | None: ...

    def prepare(self, clips: Sequence[ClipManifest], ctx: TimingContext) -> Mapping[str, float]: ...

    def describe(self, ctx: TimingContext) -> Mapping[str, str]: ...

    def close(self) -> None: ...


TIMING_TARGETS = PluginKind[type[TimingTarget]]("timing_targets")
