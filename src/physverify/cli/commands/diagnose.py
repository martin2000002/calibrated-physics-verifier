"""``physverify diagnose``: stage errors, end-to-end error, attribution and real video (ADR 0007).

Not a heavy command: ``LazyLock`` inside ``PipelineDeps`` takes the heavy lock only when a stage
has to be computed, so a fully cached diagnosis runs next to another cycle's heavy job.
"""

from __future__ import annotations

import datetime as dt
import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any

import typer

from physverify.app.context import AppContext
from physverify.cli.commands._sealed import Selection, guard_sealed
from physverify.cli.commands.extract import (
    ConsoleObserver,
    EventRecorder,
    pipeline_deps,
    resolve_dataset,
    select_clips,
    write_events,
)
from physverify.cli.context import app_context
from physverify.core.markdown import number
from physverify.diagnosis.attribution import run_attribution
from physverify.diagnosis.cases import SampledMetric
from physverify.diagnosis.checks_port import A7CheckSuite, CheckSuite
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.end_to_end import run_end_to_end
from physverify.diagnosis.real_video import run_real
from physverify.diagnosis.real_video import select_clips as real_clips
from physverify.diagnosis.report import (
    AttributionReport,
    EndToEndReport,
    Failure,
    RealVideoReport,
    Report,
    StageErrorReport,
    write_report,
)
from physverify.diagnosis.settings import MetricSettings, diagnosis_settings
from physverify.diagnosis.stage_errors import run_stage_errors
from physverify.extraction._base import StageObserver
from physverify.extraction.pipeline import ExtractionReport
from physverify.runtime.runs import new_run
from physverify.schema.clip import ClipManifest

NAME = "diagnose"
EVENTS_FILE = "events.jsonl"
ALL_DATASETS = "all"
MAX_SUMMARY_LINES = 10
STAGES, END_TO_END, ATTRIBUTION, REAL = "stages", "end_to_end", "attribution", "real"

Runner = Callable[[DiagnosisContext], Report]

app = typer.Typer(
    help="Extractor diagnosis: per-stage error, end-to-end error, attribution and real video.",
    no_args_is_help=True,
)

DatasetOption = Annotated[
    str | None, typer.Option(help="Dataset name (default: diagnosis.dataset).")
]
SubsetOption = Annotated[
    str | None, typer.Option(help="Dataset subset (default: diagnosis.subset).")
]
ClipsOption = Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")]


# --- Context ----------------------------------------------------------------------------------


def sampled_metrics(
    registry: Mapping[str, type[Any]], options: MetricSettings
) -> dict[str, SampledMetric]:
    """One instance per registered metric that also yields per-sample values."""
    instances = {key: cls(options) for key, cls in registry.items()}
    return {key: metric for key, metric in instances.items() if isinstance(metric, SampledMetric)}


def diagnosis_context(
    app: AppContext,
    observer: StageObserver,
    suite: CheckSuite | None,
    *,
    dataset: str | None = None,
    subset: str | None = None,
) -> DiagnosisContext:
    settings = app.settings
    overrides = {"dataset": dataset, "subset": subset}
    diag = diagnosis_settings(settings).model_copy(
        update={key: value for key, value in overrides.items() if value is not None}
    )
    clock = functools.partial(dt.datetime.now, dt.UTC)
    run = new_run(settings, app.invocation.argv, clock())
    return DiagnosisContext(
        settings=settings,
        diag=diag,
        resolve_dataset=functools.partial(resolve_dataset, app),
        providers=app.registry("providers"),
        metrics=sampled_metrics(app.registry("metrics"), diag.metrics),
        deps=pipeline_deps(app, observer),
        suite=suite,
        clock=clock,
        run=run,
    )


def dataset_clips(ctx: DiagnosisContext, wanted: Sequence[str]) -> list[ClipManifest]:
    return select_clips(ctx.dataset().clips(ctx.diag.subset), wanted)


def guarded_clips(ctx: DiagnosisContext, leaf: str, wanted: Sequence[str]) -> list[ClipManifest]:
    """``dataset_clips`` after the sealed-ledger guard of ``diagnose <leaf>``."""
    selected = dataset_clips(ctx, wanted)
    selection = Selection(ctx.dataset().name, ctx.diag.subset, tuple(selected))
    guard_sealed(app_context(), f"{NAME} {leaf}", [selection], run_dir=ctx.run.run_dir)
    return selected


def real_datasets(ctx: DiagnosisContext, dataset: str | None) -> tuple[str, ...]:
    return ctx.diag.real.datasets if dataset in (None, ALL_DATASETS) else (str(dataset),)


# --- Running one level ------------------------------------------------------------------------


def execute(ctx: DiagnosisContext, recorder: EventRecorder, name: str, runner: Runner) -> Report:
    """Run ``runner``, then write ``diagnosis/<name>.{json,md}`` and ``events.jsonl``."""
    run_dir = ctx.run.run_dir
    events = run_dir / EVENTS_FILE
    typer.echo(f"run: {run_dir}")
    try:
        report = runner(ctx)
    except Exception as exc:
        write_events(events, recorder.aborted_report(exc))
        raise
    write_events(events, ExtractionReport(tuple(recorder.finished), (), {}))
    _, markdown = write_report(run_dir, name, report)
    typer.echo(f"report: {markdown}")
    for line in summary_lines(report)[:MAX_SUMMARY_LINES]:
        typer.echo(line)
    return report


def _launch(
    name: str,
    runner: Runner,
    *,
    suite: bool,
    dataset: str | None = None,
    subset: str | None = None,
) -> None:
    app = app_context()
    recorder = EventRecorder(ConsoleObserver(typer.echo))
    checks = (
        A7CheckSuite.build(app.settings, app.registry("checks"), app.registry("policies"))
        if suite
        else None
    )
    ctx = diagnosis_context(app, recorder, checks, dataset=dataset, subset=subset)
    execute(ctx, recorder, name, runner)


@app.command("stages", help="Level 1: per-stage error against ground truth (error profile).")
def stages(
    dataset: DatasetOption = None,
    subset: SubsetOption = None,
    clips: ClipsOption = None,
    write_profile: Annotated[
        bool, typer.Option(help="Write the calib error profile to paths.error_profile.")
    ] = False,
    replace_mismatched: Annotated[
        bool,
        typer.Option(
            help="Replace entries measured under another profile or other params instead of "
            "refusing the merge (a deliberate re-measurement)."
        ),
    ] = False,
) -> None:
    def runner(ctx: DiagnosisContext) -> Report:
        selected = guarded_clips(ctx, "stages", clips or ())
        return run_stage_errors(
            ctx, selected, write_profile=write_profile, replace_mismatched=replace_mismatched
        )

    _launch(STAGES, runner, suite=False, dataset=dataset, subset=subset)


@app.command("end-to-end", help="Level 2: check magnitudes of the perceived chain vs the oracle.")
def end_to_end(
    dataset: DatasetOption = None, subset: SubsetOption = None, clips: ClipsOption = None
) -> None:
    def runner(ctx: DiagnosisContext) -> Report:
        return run_end_to_end(ctx, guarded_clips(ctx, "end-to-end", clips or ()))

    _launch(END_TO_END, runner, suite=True, dataset=dataset, subset=subset)


@app.command("attribution", help="Level 3: factorial attribution of check error to stages.")
def attribution(
    dataset: DatasetOption = None, subset: SubsetOption = None, clips: ClipsOption = None
) -> None:
    def runner(ctx: DiagnosisContext) -> Report:
        return run_attribution(ctx, guarded_clips(ctx, "attribution", clips or ()))

    _launch(ATTRIBUTION, runner, suite=True, dataset=dataset, subset=subset)


@app.command("real", help="Real video (end-to-end only): Physics-IQ motion, PISA falls, verdicts.")
def real(
    dataset: Annotated[
        str, typer.Option(help="physics_iq, pisa_real or all (diagnosis.real.datasets).")
    ] = ALL_DATASETS,
    subset: SubsetOption = None,
    clips: ClipsOption = None,
) -> None:
    def runner(ctx: DiagnosisContext) -> Report:
        names, subset = real_datasets(ctx, dataset), ctx.diag.subset
        chosen = real_clips(ctx, names, subset, clips or ())
        selections = [Selection(source.name, subset, found) for source, found in chosen]
        guard_sealed(app_context(), f"{NAME} real", selections, run_dir=ctx.run.run_dir)
        return run_real(ctx, names, subset, clips or ())

    _launch(REAL, runner, suite=True, subset=subset)


# --- Console summary --------------------------------------------------------------------------


def _number(value: float | None) -> str:
    return number(value, ".3g", "-", "nan")


def _failure_lines(failures: Sequence[Failure]) -> list[str]:
    return [f"FAILED {f.clip_id}: {f.error}" for f in failures]


@functools.singledispatch
def summary_lines(report: Any) -> list[str]:
    raise TypeError(f"no summary for {type(report).__name__}")


@summary_lines.register
def _stages_summary(report: StageErrorReport) -> list[str]:
    written = report.profile_path if report.profile_written else "not written"
    lines = [f"error profile: {written}", f"metric rows: {len(report.rows)}"]
    lines += [
        f"{r.stage.value}/{r.metric} {r.split.value}: n_clips {r.n_clips}" for r in report.rows
    ]
    return _failure_lines(report.failures) + lines


@summary_lines.register
def _end_to_end_summary(report: EndToEndReport) -> list[str]:
    lines = [
        f"{c.name} {row.check_id}: pairs {row.n_pairs} mae {_number(row.mae)} {row.unit}"
        f" flips {_number(row.flips.rate)}"
        for c in report.comparisons
        for row in c.checks
    ]
    return _failure_lines(report.failures) + lines


@summary_lines.register
def _attribution_summary(report: AttributionReport) -> list[str]:
    lines = [
        f"{c.check_id}: blocks {c.n_blocks} v0 {_number(c.v0)} {c.unit} shapley "
        + ", ".join(f"{stage.value} {_number(value)}" for stage, value in c.shapley.items())
        for c in report.checks
    ]
    return _failure_lines(report.failures) + lines


@summary_lines.register
def _real_summary(report: RealVideoReport) -> list[str]:
    lines = [
        f"{r.clip_id}: motion IoU {_number(r.motion_iou)}"
        f" false motion {_number(r.false_motion_rate)}"
        + (f" ({r.unavailable})" if r.unavailable else "")
        for r in report.motion
    ]
    lines += [
        f"{r.clip_id}: g [estimated] {_number(r.g_est)} m/s2 closer base {r.closer_time_base}"
        + (f" abstain {r.abstain_reason.value}" if r.abstain_reason else "")
        for r in report.falls
    ]
    lines += [
        f"{v.dataset} {v.check_id}: fail rate {_number(v.fail_rate)}" for v in report.verdicts
    ]
    return _failure_lines(report.failures) + lines
