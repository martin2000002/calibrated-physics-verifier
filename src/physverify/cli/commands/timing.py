"""``physverify timing``: the per-clip cost harness (ADR 0008).

Only runs on the H200 (``--device cuda --profile precise``) are cost measurements; any other run is
a harness check, and then no timing is printed to the terminal.
"""

from __future__ import annotations

import contextlib
import json
import signal
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Annotated, Any, cast

import typer

from physverify.app.context import AppContext
from physverify.cli.context import app_context, heavy_command
from physverify.config.models import PathsSettings, Settings
from physverify.core.errors import EXIT_ERROR, EXIT_OK, UsageError
from physverify.datasets._base import dataset_from
from physverify.extraction.weights import (
    ALLOW_PATTERNS,
    Downloader,
    NetworkDownloader,
    fetch,
    hf_repo_and_file,
    hub_cache,
)
from physverify.runtime.heavy import NullLock
from physverify.runtime.runs import RUN_FILE, RunContext, new_run
from physverify.timing._base import TimingContext, TimingTarget
from physverify.timing.budget import Bound, BudgetAnswer, BudgetQuery, answer, cost_from_results
from physverify.timing.envinfo import CudaProbe, TorchCudaProbe, run_environment, validity
from physverify.timing.generation import wan_weight_spec
from physverify.timing.harness import ClipSource, TimingHarness
from physverify.timing.hooks import brings_own_clips
from physverify.timing.records import ResultsFile, TargetStatus
from physverify.timing.report import (
    RESULTS,
    SAMPLES,
    STATS,
    SUMMARY,
    TIMING_DIR,
    CsvSampleSink,
    ResultsWriter,
    banner,
    load_results,
    load_samples,
    rebuild,
    timing_dir,
)
from physverify.timing.settings import (
    GenerationSettings,
    TargetRunSettings,
    TimingSettings,
    load_timing,
    run_settings,
    with_overrides,
)
from physverify.timing.stats import stat_rows
from physverify.timing.target_services import (
    ClipSelection,
    TimingServices,
    selected_clips,
)

NAME = "timing"
NESTED_LOCK = "held by timing run"
FAILING = frozenset({TargetStatus.FAILED, TargetStatus.PARTIAL})

Plan = list[tuple[type[TimingTarget], TargetRunSettings]]

app = typer.Typer(
    help=(
        "Per-clip cost harness. Only H200 runs (--device cuda --profile precise) are cost "
        "measurements; anything else is a harness check."
    ),
    no_args_is_help=True,
)


def _ordered(names: Sequence[str], registry: Mapping[str, Any]) -> list[str]:
    known = [name for name in names if name in registry]
    return known + sorted(name for name in registry if name not in known)


@app.command("list", help="Registered targets with their warm-up and repetitions.")
def list_targets() -> None:
    context = app_context()
    timing = load_timing(context.settings)
    registry = context.registry("timing_targets")
    typer.echo("name\trequires_gpu\twarmup\tscope\trepetitions\tmax_clips")
    for name in _ordered(timing.order, registry):
        run = run_settings(timing, name)
        limit = "-" if run.max_clips is None else run.max_clips
        typer.echo(
            f"{name}\t{registry[name].requires_gpu}\t{run.warmup}\t{run.warmup_scope}\t"
            f"{run.repetitions}\t{limit}"
        )


@app.command("env", help="Environment of a run with the current options and its validity (JSON).")
def env() -> None:
    settings = app_context().settings
    timing = load_timing(settings)
    environment = run_environment(settings, TorchCudaProbe())
    reasons = validity(settings.runtime.device, settings.profile, environment, timing.valid_profile)
    report = {"environment": environment, "valid_measurement": not reasons, "reasons": reasons}
    typer.echo(json.dumps(report, indent=2, default=str))


def fetch_generator(gen: GenerationSettings, paths: PathsSettings, downloader: Downloader) -> Path:
    """Download the pinned generator and verify it; a cut download resumes on the next call."""
    spec = wan_weight_spec(gen)
    repo, file = hf_repo_and_file(spec.ref)
    # weights.fetch skips the snapshot once model_index.json exists, which would leave missing
    # shards missing; the snapshot itself skips the files already complete.
    downloader.hf_snapshot(repo, spec.revision, hub_cache(paths), (*ALLOW_PATTERNS, file))
    return fetch(spec, paths, downloader)


@app.command("fetch", help="Download the pinned Wan2.1 weights (about 29 GB; for the H200).")
@heavy_command
def fetch_weights() -> None:
    settings = app_context().settings
    typer.echo(f"hub cache: {hub_cache(settings.paths)}")
    path = fetch_generator(load_timing(settings).generation, settings.paths, NetworkDownloader())
    typer.echo(f"verified {path}")


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


def _services(context: AppContext, selection: ClipSelection, probe: CudaProbe) -> TimingServices:
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


def _start_results(
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


def _time_targets(
    plan: Plan,
    services: TimingServices,
    clips: ClipSource,
    ctx: TimingContext,
    writer: ResultsWriter,
) -> bool:
    """Run the harness; False when it was interrupted (what it wrote is kept)."""
    with CsvSampleSink(timing_dir(ctx.run_dir)) as sink, sigterm_as_interrupt():
        harness = TimingHarness(sink, on_outcome=writer.record)
        try:
            harness.run(plan, lambda cls: _construct(cls, services), clips, ctx)
        except KeyboardInterrupt:
            typer.echo("interrupted: the samples written so far are kept", err=True)
            return False
    return True


def _report(results: ResultsFile, directory: Path) -> None:
    """Terminal output: a harness check prints no numbers, only the banner, statuses and paths."""
    head = banner(results)
    if head is not None:
        typer.echo(head)
    for target in results.targets:
        reason = f" ({target.reason})" if target.reason else ""
        typer.echo(f"{target.name}: {target.status.value}{reason}")
    for name in (RESULTS, SAMPLES, STATS, SUMMARY):
        typer.echo(f"{name}: {directory / name}")
    if results.valid_measurement:
        typer.echo((directory / SUMMARY).read_text(encoding="utf-8"))


def _exit_code(results: ResultsFile, finished: bool) -> int:
    failing = any(target.status in FAILING for target in results.targets)
    return EXIT_ERROR if failing or not finished else EXIT_OK


@app.command("run", help="Time targets; only an H200 run (cuda, precise profile) is a measurement.")
def run(
    target: Annotated[
        list[str] | None, typer.Option(help="Target name (repeatable); default: timing.order.")
    ] = None,
    dataset: Annotated[
        str | None, typer.Option(help="Dataset for targets that do not bring their own clips.")
    ] = None,
    subset: Annotated[str | None, typer.Option(help="Dataset subset.")] = None,
    clips: Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Keep the first N clips.")] = None,
    repetitions: Annotated[
        int | None, typer.Option(help="Timed calls per clip, for every selected target.")
    ] = None,
    warmup: Annotated[
        int | None, typer.Option(help="Warm-up calls, for every selected target.")
    ] = None,
) -> None:
    context = app_context()
    settings = context.settings
    timing = load_timing(settings)
    plan = plan_targets(
        context.registry("timing_targets"),
        timing,
        target or [],
        warmup=warmup,
        repetitions=repetitions,
    )
    probe = TorchCudaProbe()
    check_device(plan, settings.runtime.device, probe)
    selection = ClipSelection(dataset, subset, tuple(clips or ()), limit)
    services = _services(context, selection, probe)
    chosen = clip_source(plan, services, settings)
    with context.heavy_lock():
        run_context = new_run(context.settings, context.invocation.argv)
        typer.echo(f"run: {run_context.run_dir}")
        writer = _start_results(run_context, settings, timing, probe)
        runtime = settings.runtime
        ctx = TimingContext(settings, runtime.device, runtime.threads, run_context.run_dir)
        finished = _time_targets(plan, services, chosen, ctx, writer)
        directory = timing_dir(run_context.run_dir)
        results = rebuild(directory, timing.max_iqr_rel, timing.total_measures)
    _report(results, directory)
    code = _exit_code(results, finished)
    if code != EXIT_OK:
        raise typer.Exit(code=code)


def _timing_directory(run_dir: Path) -> Path:
    for candidate in (run_dir, run_dir / TIMING_DIR):
        if (candidate / RESULTS).is_file():
            return candidate
    raise UsageError(f"{run_dir} is not a timing run (no {RESULTS} in it or in {TIMING_DIR}/)")


RunOption = Annotated[Path, typer.Option("--run", help="Run directory (or its timing/ folder).")]


@app.command("summarize", help="Rebuild stats.csv and summary.md from samples.csv.")
def summarize(run_dir: RunOption) -> None:
    timing = load_timing(app_context().settings)
    directory = _timing_directory(run_dir)
    _report(rebuild(directory, timing.max_iqr_rel, timing.total_measures), directory)


def _bound_line(label: str, result: Bound) -> str:
    return (
        f"{label}: {result.cost_s:.4g} s per video; available {result.available_s:.6g} s; "
        f"training prompts {result.prompts}; preference pairs {result.pairs_min}-"
        f"{result.pairs_max} (before abstentions)"
    )


def budget_lines(result: BudgetAnswer, query: BudgetQuery) -> list[str]:
    extract_load = (
        f"included with B_x = {query.extract_batch}"
        if result.extract_load_included
        else "omitted (pass --extract-batch to include it)"
    )
    generator_load = (
        "not measured"
        if result.generator_load_s is None
        else f"{result.generator_load_s:.4g} s once per job, not in the cost per video"
    )
    return [
        f"budget: H = {query.gpu_hours:g} GPU hours, fine-tuning {query.finetune_hours:g} h, "
        f"K = {query.candidates}, Q = {query.eval_prompts}, K' = {query.eval_samples}, "
        f"B = {query.batch_size}",
        _bound_line("sequential c_seq = g_B + e + v + L_x/B_x", result.sequential),
        _bound_line("generator only c_gen = g_B", result.generator_only),
        f"extraction load L_x/B_x: {extract_load}",
        f"generator load L_g: {generator_load}",
    ]


@app.command(
    "budget", help="Training prompts and pairs that fit in a GPU budget (valid runs only)."
)
def budget(
    run_dir: RunOption,
    gpu_hours: Annotated[float, typer.Option(help="GPU hours available (H).")],
    candidates: Annotated[int, typer.Option(help="Candidate videos per prompt (K).")],
    eval_prompts: Annotated[int, typer.Option(help="Evaluation prompts (Q).")] = 0,
    eval_samples: Annotated[int, typer.Option(help="Samples per prompt, trained model (K').")] = 1,
    finetune_hours: Annotated[float, typer.Option(help="GPU hours of fine-tuning (T_ft).")] = 0.0,
    extract_batch: Annotated[
        int | None, typer.Option(help="Clips per extraction process (B_x).")
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Generation batch size (B).")] = 1,
) -> None:
    query = BudgetQuery(
        gpu_hours, candidates, eval_prompts, eval_samples, finetune_hours, extract_batch, batch_size
    )
    timing = load_timing(app_context().settings)
    directory = _timing_directory(run_dir)
    results = load_results(directory)
    rows = stat_rows(load_samples(directory), timing.max_iqr_rel)
    cost = cost_from_results(results, rows, timing.total_measures, query.batch_size)
    for line in budget_lines(answer(cost, query), query):
        typer.echo(line)
