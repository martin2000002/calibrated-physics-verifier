"""Wiring of a diagnosis run from the application context."""

from __future__ import annotations

import datetime as dt
import functools
from collections.abc import Callable, Sequence

from physverify.app.context import AppContext
from physverify.app.extraction import recorded_run
from physverify.app.sealing import Selection, guard_sealed
from physverify.app.services import check_suite, dataset_for, pipeline_deps
from physverify.core.output import Echo
from physverify.datasets._base import select_clips
from physverify.diagnosis.attribution_run import run_attribution
from physverify.diagnosis.context import DiagnosisContext, sampled_metrics
from physverify.diagnosis.end_to_end import run_end_to_end
from physverify.diagnosis.real_video import run_real
from physverify.diagnosis.real_video import select_clips as real_clips
from physverify.diagnosis.report import Report
from physverify.diagnosis.report_markdown import write_report
from physverify.diagnosis.settings import diagnosis_settings
from physverify.diagnosis.stage_errors import run_stage_errors
from physverify.diagnosis.summary import MAX_SUMMARY_LINES, summary_lines
from physverify.diagnosis.verification_adapter import CheckSuite
from physverify.extraction._base import StageObserver
from physverify.extraction.events import ConsoleObserver, EventRecorder
from physverify.extraction.pipeline import ExtractionReport
from physverify.runtime.runs import new_run
from physverify.schema.clip import ClipManifest

EVENTS_FILE = "events.jsonl"
ALL_DATASETS = "all"

Runner = Callable[[DiagnosisContext], Report]


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
        dataset_for=functools.partial(dataset_for, app),
        providers=app.registry("providers"),
        checks=app.registry("checks"),
        metrics=sampled_metrics(app.registry("metrics"), diag.metrics),
        deps=pipeline_deps(app, observer),
        suite=suite,
        clock=clock,
        run=run,
    )


def execute(
    ctx: DiagnosisContext,
    recorder: EventRecorder,
    name: str,
    runner: Runner,
    echo: Echo,
) -> Report:
    """Run ``runner``, then write ``diagnosis/<name>.{json,md}`` and ``events.jsonl``."""
    report = recorded_run(
        ctx.run.run_dir,
        EVENTS_FILE,
        recorder,
        functools.partial(runner, ctx),
        lambda _: ExtractionReport(tuple(recorder.finished), (), {}),
        echo,
    )
    _, markdown = write_report(ctx.run.run_dir, name, report)
    echo(f"report: {markdown}")
    for line in summary_lines(report)[:MAX_SUMMARY_LINES]:
        echo(line)
    return report


def launch(
    app: AppContext,
    name: str,
    runner: Runner,
    echo: Echo,
    *,
    suite: bool,
    dataset: str | None = None,
    subset: str | None = None,
) -> None:
    recorder = EventRecorder(ConsoleObserver(echo))
    checks = check_suite(app) if suite else None
    ctx = diagnosis_context(app, recorder, checks, dataset=dataset, subset=subset)
    execute(ctx, recorder, name, runner, echo)


def dataset_clips(ctx: DiagnosisContext, wanted: Sequence[str]) -> list[ClipManifest]:
    return select_clips(ctx.dataset().clips(ctx.diag.subset), wanted)


def guarded_clips(
    app: AppContext, ctx: DiagnosisContext, command: str, wanted: Sequence[str], err: Echo
) -> list[ClipManifest]:
    """``dataset_clips`` after the sealed-ledger guard of ``command``."""
    selected = dataset_clips(ctx, wanted)
    selection = Selection(ctx.dataset().name, ctx.diag.subset, tuple(selected))
    guard_sealed(app, command, [selection], err=err, run_dir=ctx.run.run_dir)
    return selected


def real_datasets(ctx: DiagnosisContext, dataset: str | None) -> tuple[str, ...]:
    return ctx.diag.real.datasets if dataset in (None, ALL_DATASETS) else (str(dataset),)


def stage_errors_runner(
    app: AppContext,
    command: str,
    clips: Sequence[str],
    *,
    write_profile: bool,
    replace_mismatched: bool,
    err: Echo,
) -> Runner:
    def runner(ctx: DiagnosisContext) -> Report:
        selected = guarded_clips(app, ctx, command, clips, err)
        return run_stage_errors(
            ctx, selected, write_profile=write_profile, replace_mismatched=replace_mismatched
        )

    return runner


def end_to_end_runner(app: AppContext, command: str, clips: Sequence[str], *, err: Echo) -> Runner:
    def runner(ctx: DiagnosisContext) -> Report:
        return run_end_to_end(ctx, guarded_clips(app, ctx, command, clips, err))

    return runner


def attribution_runner(app: AppContext, command: str, clips: Sequence[str], *, err: Echo) -> Runner:
    def runner(ctx: DiagnosisContext) -> Report:
        return run_attribution(ctx, guarded_clips(app, ctx, command, clips, err))

    return runner


def real_runner(
    app: AppContext, command: str, dataset: str, clips: Sequence[str], *, err: Echo
) -> Runner:
    def runner(ctx: DiagnosisContext) -> Report:
        names, subset = real_datasets(ctx, dataset), ctx.diag.subset
        chosen = real_clips(ctx, names, subset, clips)
        selections = [Selection(source.name, subset, found) for source, found in chosen]
        guard_sealed(app, command, selections, err=err, run_dir=ctx.run.run_dir)
        return run_real(ctx, names, subset, clips)

    return runner
