"""Per-stage spans of the extraction pipeline, taken from A5's StageObserver hook.

``wall_s`` and ``peak_rss_mb`` are copied from each ``StageEvent`` untouched (A5 times only the
provider call). The recorder adds what the event lacks: the ``on_stage_start -> on_stage_end`` span
(model load on a stage's first uncached cell, upstream cache loads and the cache store), the
process CPU time and the CUDA peak.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from physverify.extraction._base import StageEvent, StageObserver
from physverify.timing.records import MIB


class CudaMemory(Protocol):
    def reset(self) -> None: ...

    def peak_mb(self) -> float | None: ...  # None: no CUDA, or torch not loaded


def loaded_cuda_torch() -> Any | None:
    """torch when this process already imported it and CUDA is usable; never imports it."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    return torch


class TorchCudaMemory:
    """Peak memory reserved by torch's caching allocator on ``device``; never imports torch.

    Reserved (not allocated) is what the process holds on the GPU; the CUDA context itself is
    not included."""

    def __init__(self, device: str = "cuda") -> None:
        self._device = device

    def reset(self) -> None:
        torch = loaded_cuda_torch()
        if torch is not None:
            torch.cuda.reset_peak_memory_stats(self._device)

    def peak_mb(self) -> float | None:
        torch = loaded_cuda_torch()
        if torch is None:
            return None
        return float(torch.cuda.max_memory_reserved(self._device)) / MIB


class NoCudaMemory:
    def reset(self) -> None:
        return None

    def peak_mb(self) -> float | None:
        return None


def memory_for(device: str) -> CudaMemory:
    return TorchCudaMemory(device) if device.startswith("cuda") else NoCudaMemory()


@dataclass(frozen=True)
class StageSpan:
    clip_id: str
    stage: str
    provider: str
    device: str
    wall_s: float  # from StageEvent
    peak_rss_mb: float  # from StageEvent
    span_s: float
    cpu_s: float  # process CPU over all threads, so it can exceed span_s
    cuda_peak_mb: float | None


@dataclass(frozen=True)
class _Start:
    wall: float
    cpu: float


class StageRecorder:
    """A StageObserver; cached cells are ignored and a start without an end is dropped."""

    def __init__(
        self,
        cuda: CudaMemory,
        clock: Callable[[], float] = time.perf_counter,
        cpu_clock: Callable[[], float] = time.process_time,
    ) -> None:
        self._cuda = cuda
        self._clock = clock
        self._cpu_clock = cpu_clock
        self._open: dict[tuple[str, str], _Start] = {}
        self._spans: list[StageSpan] = []

    def on_stage_start(self, event: StageEvent) -> None:
        key = (event.clip_id, event.stage.value)
        self._open.pop(key, None)
        if event.cached:
            return
        self._cuda.reset()
        self._open[key] = _Start(self._clock(), self._cpu_clock())

    def on_stage_end(self, event: StageEvent) -> None:
        start = self._open.pop((event.clip_id, event.stage.value), None)
        if event.cached or start is None:
            return
        self._spans.append(self._span(event, start))

    def _span(self, event: StageEvent, start: _Start) -> StageSpan:
        return StageSpan(
            clip_id=event.clip_id,
            stage=event.stage.value,
            provider=event.provider,
            device=event.device,
            wall_s=event.wall_s,
            peak_rss_mb=event.peak_rss_mb,
            span_s=self._clock() - start.wall,
            cpu_s=self._cpu_clock() - start.cpu,
            cuda_peak_mb=self._cuda.peak_mb(),
        )

    @property
    def spans(self) -> tuple[StageSpan, ...]:
        return tuple(self._spans)


class TeeObserver:
    """Forwards every stage event to several observers, in order (``None`` entries are skipped)."""

    def __init__(self, *observers: StageObserver | None) -> None:
        self._observers = tuple(observer for observer in observers if observer is not None)

    def on_stage_start(self, event: StageEvent) -> None:
        for observer in self._observers:
            observer.on_stage_start(event)

    def on_stage_end(self, event: StageEvent) -> None:
        for observer in self._observers:
            observer.on_stage_end(event)
