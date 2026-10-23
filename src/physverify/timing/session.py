"""One timing session: the targets to time, the device and clip checks, the harness run under
SIGTERM, and the terminal lines of its results."""

from __future__ import annotations

import contextlib
import json
import signal
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any, Final, cast

from physverify.config.models import Settings
from physverify.core.errors import EXIT_ERROR, EXIT_OK, UsageError
from physverify.datasets._base import dataset_from
from physverify.runtime.runs import RUN_FILE, RunContext
from physverify.timing._base import TimingContext, TimingTarget
from physverify.timing.environment import CudaProbe, run_environment, validity
from physverify.timing.harness import ClipSource, TimingHarness
from physverify.timing.hooks import brings_own_clips
from physverify.timing.records import ResultsFile, TargetStatus
from physverify.timing.report import SUMMARY, banner, timing_dir
from physverify.timing.results_io import RESULTS, SAMPLES, STATS, CsvSampleSink, ResultsWriter
from physverify.timing.settings import (
    TargetRunSettings,
    TimingSettings,
    run_settings,
    with_overrides,
)
from physverify.timing.target_services import TimingServices, selected_clips

FAILING: Final = frozenset({TargetStatus.FAILED, TargetStatus.PARTIAL})

Plan = list[tuple[type[TimingTarget], TargetRunSettings]]


def ordered(names: Sequence[str], registry: Mapping[str, Any]) -> list[str]:
    known = [name for name in names if name in registry]
    return known + sorted(name for name in registry if name not in known)


def plan_targets(
    registry: Mapping[str, type[TimingTarget]],
    timing: TimingSettings,
    names: Sequence[str],
    *,
    warmup: int | None,
    repetitions: int | None,
) -> Plan:
    chosen = list(dict.fromkeys(names or timing.order))
    unknown = [name for name in chosen if name not in registry]
    if unknown:
        raise UsageError(f"unknown timing target {unknown}; known: {sorted(registry)}")
    return [
        (
            registry[name],
            with_overrides(run_settings(timing, name), warmup=warmup, repetitions=repetitions),
        )
        for name in chosen
    ]


def check_device(plan: Plan, device: str, probe: CudaProbe) -> None:
    """D6: a GPU target needs ``--device cuda``, and ``--device cuda`` needs a CUDA host."""
    gpu_targets = [cls.name for cls, _ in plan if cls.requires_gpu]
    on_cuda = device.startswith("cuda")
    if gpu_targets and not on_cuda:
        raise UsageError(
            f"{', '.join(gpu_targets)} requires --device cuda (device is {device!r}); "
            "generation runs only on the H200. Pick CPU targets with --target."
        )
    if on_cuda and not probe.available():
        raise UsageError(f"--device {device}: CUDA is not available on this host")


def _needs_selection(plan: Plan) -> list[str]:
    return [cls.name for cls, _ in plan if not brings_own_clips(cls)]


def clip_source(plan: Plan, services: TimingServices, settings: Settings) -> ClipSource:
    """Checks ``--dataset`` before anything is created (usage errors exit 2) but reads the clips
    only when a target asks: an earlier target may create them (``wan21_generate`` writes the
    ``local_video`` subset that the extraction and verification targets then time)."""
    needing = _needs_selection(plan)
    if not needing:
        return tuple
    if services.selection.dataset is None:
        raise UsageError(f"--dataset is required by {', '.join(needing)}")
    dataset_from(services.datasets, services.selection.dataset, settings)
    return lambda: selected_clips(services, settings)


def _construct(cls: type[TimingTarget], services: TimingServices) -> TimingTarget:
    factory: Any = cls  # targets take the services; the frozen protocol does not declare __init__
    return cast(TimingTarget, factory(services))


@contextlib.contextmanager
def sigterm_as_interrupt() -> Iterator[None]:
    """Turn SIGTERM (a cluster time limit) into KeyboardInterrupt so targets close and the files
    are finished; restores the previous handler."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def start_results(
    run_context: RunContext, settings: Settings, timing: TimingSettings, probe: CudaProbe
) -> ResultsWriter:
    environment = run_environment(settings, probe)
    reasons = validity(settings.runtime.device, settings.profile, environment, timing.valid_profile)
    record = json.loads((run_context.run_dir / RUN_FILE).read_text(encoding="utf-8"))
    results = ResultsFile(
        valid_measurement=not reasons,
        validity_reasons=reasons,
        run=record,
        environment=environment,
    )
    return ResultsWriter(timing_dir(run_context.run_dir), results)


def time_targets(
    plan: Plan,
    services: TimingServices,
    clips: ClipSource,
    ctx: TimingContext,
    writer: ResultsWriter,
    *,
    err: Callable[[str], None],
) -> bool:
    """Run the harness; False when it was interrupted (what it wrote is kept)."""
    with CsvSampleSink(timing_dir(ctx.run_dir)) as sink, sigterm_as_interrupt():
        harness = TimingHarness(sink, on_outcome=writer.record)
        try:
            harness.run(plan, lambda cls: _construct(cls, services), clips, ctx)
        except KeyboardInterrupt:
            err("interrupted: the samples written so far are kept")
            return False
    return True


def report(results: ResultsFile, directory: Path, echo: Callable[[str], None]) -> None:
    """Terminal output: a harness check prints no numbers, only the banner, statuses and paths."""
    head = banner(results)
    if head is not None:
        echo(head)
    for target in results.targets:
        reason = f" ({target.reason})" if target.reason else ""
        echo(f"{target.name}: {target.status.value}{reason}")
    for name in (RESULTS, SAMPLES, STATS, SUMMARY):
        echo(f"{name}: {directory / name}")
    if results.valid_measurement:
        echo((directory / SUMMARY).read_text(encoding="utf-8"))


def exit_code(results: ResultsFile, finished: bool) -> int:
    failing = any(target.status in FAILING for target in results.targets)
    return EXIT_ERROR if failing or not finished else EXIT_OK
