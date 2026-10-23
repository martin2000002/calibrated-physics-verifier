"""Timing wiring: the services a timing run hands its targets, and the run, list, env, summarize
and budget services behind ``physverify timing``."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from physverify.app.context import AppContext
from physverify.checks._base import CHECKS
from physverify.config.models import Settings
from physverify.config.plugin_context import registry_of
from physverify.core.output import Echo
from physverify.datasets._base import DATASETS
from physverify.extraction._base import PROVIDERS
from physverify.extraction.weights import NetworkDownloader, hub_cache
from physverify.generation.wan21 import fetch_generator
from physverify.runtime.heavy import NullLock
from physverify.runtime.runs import new_run
from physverify.timing._base import TIMING_TARGETS, TimingContext
from physverify.timing.budget import BudgetQuery as BudgetQuery
from physverify.timing.budget import answer, budget_lines, cost_from_results
from physverify.timing.environment import CudaProbe, TorchCudaProbe, run_environment, validity
from physverify.timing.report import rebuild, run_lines, timing_dir
from physverify.timing.results_io import load_results, load_samples, timing_directory
from physverify.timing.session import (
    check_device,
    clip_source,
    exit_code,
    ordered,
    plan_targets,
    start_results,
    time_targets,
)
from physverify.timing.settings import load_timing, run_settings
from physverify.timing.stats import stat_rows
from physverify.timing.target_services import ClipSelection as ClipSelection
from physverify.timing.target_services import TimingServices
from physverify.verification._base import POLICIES

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
        datasets=registry_of(context, DATASETS),
        providers=registry_of(context, PROVIDERS),
        checks=registry_of(context, CHECKS),
        policies=registry_of(context, POLICIES),
        lock_factory=lambda: NullLock(NESTED_LOCK),
        docker=context.docker,
        cuda=probe,
        selection=selection,
    )


def target_lines(context: AppContext) -> list[str]:
    timing = load_timing(context.settings)
    registry = registry_of(context, TIMING_TARGETS)
    lines = ["name\trequires_gpu\twarmup\tscope\trepetitions\tmax_clips"]
    for name in ordered(timing.order, registry):
        run = run_settings(timing, name)
        limit = "-" if run.max_clips is None else run.max_clips
        lines.append(
            f"{name}\t{registry[name].requires_gpu}\t{run.warmup}\t{run.warmup_scope}\t"
            f"{run.repetitions}\t{limit}"
        )
    return lines


def environment_report(settings: Settings) -> str:
    timing = load_timing(settings)
    environment = run_environment(settings, TorchCudaProbe())
    reasons = validity(settings.runtime.device, settings.profile, environment, timing.valid_profile)
    payload = {"environment": environment, "valid_measurement": not reasons, "reasons": reasons}
    return json.dumps(payload, indent=2, default=str)


def run_timing(
    context: AppContext,
    request: TimingRequest,
    *,
    echo: Echo,
    err: Echo,
) -> int:
    """Time the requested targets under the heavy lock; returns the exit code."""
    settings = context.settings
    timing = load_timing(settings)
    plan = plan_targets(
        registry_of(context, TIMING_TARGETS),
        timing,
        request.targets,
        warmup=request.warmup,
        repetitions=request.repetitions,
    )
    probe = TorchCudaProbe()
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
    for line in run_lines(results, directory):
        echo(line)
    return exit_code(results, finished)


def fetch_weights(settings: Settings, *, echo: Echo) -> None:
    """Download and verify the pinned generator weights into the hub cache."""
    echo(f"hub cache: {hub_cache(settings.paths)}")
    path = fetch_generator(load_timing(settings).generation, settings.paths, NetworkDownloader())
    echo(f"verified {path}")


def summarize(settings: Settings, run_dir: Path, *, echo: Echo) -> None:
    timing = load_timing(settings)
    directory = timing_directory(run_dir)
    results = rebuild(directory, timing.max_iqr_rel, timing.total_measures)
    for line in run_lines(results, directory):
        echo(line)


def budget_report(settings: Settings, run_dir: Path, query: BudgetQuery) -> list[str]:
    timing = load_timing(settings)
    directory = timing_directory(run_dir)
    results = load_results(directory)
    rows = stat_rows(load_samples(directory), timing.max_iqr_rel)
    cost = cost_from_results(results, rows, timing.total_measures, query.batch_size)
    return budget_lines(answer(cost, query), query)
