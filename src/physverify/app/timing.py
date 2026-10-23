"""Timing wiring: the services a timing run hands its targets, and the run, list, env, summarize
and budget services behind ``physverify timing``."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from physverify.app.context import AppContext
from physverify.config.models import Settings
from physverify.runtime.heavy import NullLock
from physverify.runtime.runs import new_run
from physverify.timing._base import TimingContext
from physverify.timing.budget import BudgetQuery, answer, budget_lines, cost_from_results
from physverify.timing.environment import CudaProbe, run_environment, validity
from physverify.timing.report import rebuild, timing_dir
from physverify.timing.results_io import load_results, load_samples, timing_directory
from physverify.timing.session import (
    check_device,
    clip_source,
    exit_code,
    ordered,
    plan_targets,
    report,
    start_results,
    time_targets,
)
from physverify.timing.settings import load_timing, run_settings
from physverify.timing.stats import stat_rows
from physverify.timing.target_services import ClipSelection, TimingServices

NESTED_LOCK: Final = "held by timing run"


@dataclass(frozen=True)
class TimingRequest:
    """``timing run``'s options: the targets, the clips and the per-target overrides."""

    targets: Sequence[str]
    selection: ClipSelection
    warmup: int | None
    repetitions: int | None


def timing_services(
    context: AppContext, selection: ClipSelection, probe: CudaProbe
) -> TimingServices:
    return TimingServices(
        datasets=context.registry("datasets"),
        providers=context.registry("providers"),
        checks=context.registry("checks"),
        policies=context.registry("policies"),
        lock_factory=lambda: NullLock(NESTED_LOCK),
        docker=context.docker,
        cuda=probe,
        selection=selection,
    )


def target_lines(context: AppContext) -> list[str]:
    timing = load_timing(context.settings)
    registry = context.registry("timing_targets")
    lines = ["name\trequires_gpu\twarmup\tscope\trepetitions\tmax_clips"]
    for name in ordered(timing.order, registry):
        run = run_settings(timing, name)
        limit = "-" if run.max_clips is None else run.max_clips
        lines.append(
            f"{name}\t{registry[name].requires_gpu}\t{run.warmup}\t{run.warmup_scope}\t"
            f"{run.repetitions}\t{limit}"
        )
    return lines


def environment_report(settings: Settings, make_probe: Callable[[], CudaProbe]) -> str:
    timing = load_timing(settings)
    environment = run_environment(settings, make_probe())
    reasons = validity(settings.runtime.device, settings.profile, environment, timing.valid_profile)
    payload = {"environment": environment, "valid_measurement": not reasons, "reasons": reasons}
    return json.dumps(payload, indent=2, default=str)


def run_timing(
    context: AppContext,
    request: TimingRequest,
    make_probe: Callable[[], CudaProbe],
    *,
    echo: Callable[[str], None],
    err: Callable[[str], None],
) -> int:
    """Time the requested targets under the heavy lock; returns the exit code."""
    settings = context.settings
    timing = load_timing(settings)
    plan = plan_targets(
        context.registry("timing_targets"),
        timing,
        request.targets,
        warmup=request.warmup,
        repetitions=request.repetitions,
    )
    probe = make_probe()
    check_device(plan, settings.runtime.device, probe)
    services = timing_services(context, request.selection, probe)
    chosen = clip_source(plan, services, settings)
    with context.heavy_lock():
        run_context = new_run(context.settings, context.invocation.argv)
        echo(f"run: {run_context.run_dir}")
        writer = start_results(run_context, settings, timing, probe)
        runtime = settings.runtime
        ctx = TimingContext(settings, runtime.device, runtime.threads, run_context.run_dir)
        finished = time_targets(plan, services, chosen, ctx, writer, err=err)
        directory = timing_dir(run_context.run_dir)
        results = rebuild(directory, timing.max_iqr_rel, timing.total_measures)
    report(results, directory, echo)
    return exit_code(results, finished)


def summarize(settings: Settings, run_dir: Path, echo: Callable[[str], None]) -> None:
    timing = load_timing(settings)
    directory = timing_directory(run_dir)
    report(rebuild(directory, timing.max_iqr_rel, timing.total_measures), directory, echo)


def budget_report(settings: Settings, run_dir: Path, query: BudgetQuery) -> list[str]:
    timing = load_timing(settings)
    directory = timing_directory(run_dir)
    results = load_results(directory)
    rows = stat_rows(load_samples(directory), timing.max_iqr_rel)
    cost = cost_from_results(results, rows, timing.total_measures, query.batch_size)
    return budget_lines(answer(cost, query), query)
