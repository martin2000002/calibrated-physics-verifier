"""What timing targets receive from the command: registries, lock, container runner, clip
selection, and the clocks and device hooks (injected so tests never need a GPU or torch)."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass

from physverify.checks._base import Check
from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.datasets._base import DatasetSource, select_clips
from physverify.extraction._base import StageProvider
from physverify.extraction.pipeline_resources import ProcRssProbe, RssProbe
from physverify.schema.clip import ClipManifest
from physverify.synth.store import ContainerRunner
from physverify.timing.envinfo import CudaProbe
from physverify.timing.recorder import CudaMemory, StageRecorder, loaded_cuda_torch, memory_for
from physverify.verification._base import VerdictPolicy


def cuda_sync(device: str) -> None:
    """Wait for the kernels queued on ``device``; a no-op off CUDA or before torch is loaded."""
    torch = loaded_cuda_torch()
    if torch is not None and device.startswith("cuda"):
        torch.cuda.synchronize(device)


def cuda_empty_cache() -> None:
    torch = loaded_cuda_torch()
    if torch is not None:
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class ClipSelection:
    """``--dataset/--subset/--clips/--limit`` of ``timing run``."""

    dataset: str | None
    subset: str | None = None
    clip_ids: tuple[str, ...] = ()
    limit: int | None = None


@dataclass(frozen=True)
class TimingServices:
    datasets: Mapping[str, type[DatasetSource]]
    providers: Mapping[str, type[StageProvider]]
    checks: Mapping[str, type[Check]]
    policies: Mapping[str, type[VerdictPolicy]]
    lock_factory: Callable[[], AbstractContextManager[object]]  # a NullLock inside `timing run`
    docker: Callable[[], ContainerRunner]
    cuda: CudaProbe
    selection: ClipSelection
    clock: Callable[[], float] = time.perf_counter
    cpu_clock: Callable[[], float] = time.process_time
    memory: Callable[[str], CudaMemory] = memory_for
    sync: Callable[[str], None] = cuda_sync
    empty_cache: Callable[[], None] = cuda_empty_cache
    rss: Callable[[], RssProbe] = ProcRssProbe

    def recorder(self, device: str) -> StageRecorder:
        return StageRecorder(self.memory(device), self.clock, self.cpu_clock)


def dataset_source(services: TimingServices, settings: Settings, name: str) -> DatasetSource:
    if name not in services.datasets:
        raise UsageError(f"unknown dataset {name!r}; known: {sorted(services.datasets)}")
    return services.datasets[name](settings)


def selected_clips(services: TimingServices, settings: Settings) -> list[ClipManifest]:
    """The command-line selection, for targets that do not bring their own clips."""
    selection = services.selection
    if selection.dataset is None:
        raise UsageError("--dataset is required by targets that do not bring their own clips")
    source = dataset_source(services, settings, selection.dataset)
    clips = select_clips(source.clips(selection.subset), selection.clip_ids)
    return clips if selection.limit is None else clips[: selection.limit]
