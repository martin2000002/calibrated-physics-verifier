"""Timing target contract (wall time, CPU, peak memory of a workload)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from physverify.config.models import Settings
from physverify.extraction._base import StageObserver
from physverify.schema.clip import ClipManifest


@dataclass(frozen=True)
class TimingContext:
    settings: Settings
    device: str
    threads: int
    run_dir: Path
    observer: StageObserver | None = None


@runtime_checkable
class TimingTarget(Protocol):
    """Registered by ``name``; returns named measurements (seconds, MB)."""

    name: ClassVar[str]
    requires_gpu: ClassVar[bool]

    def run(self, clip: ClipManifest, ctx: TimingContext) -> Mapping[str, float]: ...
