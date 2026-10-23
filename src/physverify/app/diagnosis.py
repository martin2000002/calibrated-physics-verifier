"""Wiring of a diagnosis run from the application context."""

from __future__ import annotations

import datetime as dt
import functools
from collections.abc import Callable

from physverify.app.context import AppContext
from physverify.app.extraction import recorded_run
from physverify.app.services import check_suite, dataset_for, pipeline_deps
from physverify.diagnosis.context import DiagnosisContext, sampled_metrics
from physverify.diagnosis.report import Report
from physverify.diagnosis.report_markdown import write_report
from physverify.diagnosis.settings import diagnosis_settings
from physverify.diagnosis.summary import MAX_SUMMARY_LINES, summary_lines
from physverify.diagnosis.verification_adapter import CheckSuite
from physverify.extraction._base import StageObserver
from physverify.extraction.events import ConsoleObserver, EventRecorder
from physverify.extraction.pipeline import ExtractionReport
from physverify.runtime.runs import new_run

EVENTS_FILE = "events.jsonl"

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
    echo: Callable[[str], None],
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
    echo: Callable[[str], None],
    *,
    suite: bool,
    dataset: str | None = None,
    subset: str | None = None,
) -> None:
    recorder = EventRecorder(ConsoleObserver(echo))
    checks = check_suite(app) if suite else None
    ctx = diagnosis_context(app, recorder, checks, dataset=dataset, subset=subset)
    execute(ctx, recorder, name, runner, echo)
