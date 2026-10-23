"""An extraction run from the application context, with its stage events kept on disk."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.services import dataset_for, pipeline_deps, plan_for
from physverify.core.output import Echo
from physverify.datasets._base import select_clips
from physverify.extraction.events import (
    ConsoleObserver,
    EventRecorder,
    summary_lines,
    write_events,
)
from physverify.extraction.pipeline import ExtractionPipeline, ExtractionReport
from physverify.runtime.runs import new_run
from physverify.schema.stages import parse_stages

EVENTS_FILE = "extraction.jsonl"


def recorded_run[T](
    run_dir: Path,
    events_file: str,
    recorder: EventRecorder,
    run: Callable[[], T],
    report_of: Callable[[T], ExtractionReport],
    echo: Echo,
) -> T:
    """Print the run directory, call ``run`` and write its stage events to ``events_file``, also
    when it aborts."""
    events = run_dir / events_file
    echo(f"run: {run_dir}")
    try:
        result = run()
    except Exception as exc:
        write_events(events, recorder.aborted_report(exc))
        raise
    write_events(events, report_of(result))
    return result


def run_extraction(
    context: AppContext,
    dataset: str,
    clips: Sequence[str],
    gt: Sequence[str],
    *,
    fps: float | None,
    max_side: int | None,
    subset: str | None,
    echo: Echo,
) -> ExtractionReport:
    """Run the extraction plan over the selected clips and print its per-stage summary."""
    source = dataset_for(context, dataset)
    selected = select_clips(source.clips(subset), clips)
    plan = plan_for(
        context,
        dataset=dataset,
        gt_stages=parse_stages(gt),
        fps=fps,
        max_side=max_side,
    )
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    recorder = EventRecorder(ConsoleObserver(echo))

    def run() -> ExtractionReport:
        return ExtractionPipeline(plan, pipeline_deps(context, recorder)).run(source, selected)

    report = recorded_run(run_dir, EVENTS_FILE, recorder, run, _same, echo)
    for line in summary_lines(report):
        echo(line)
    return report


def _same(report: ExtractionReport) -> ExtractionReport:
    return report
