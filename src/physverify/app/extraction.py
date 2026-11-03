"""An extraction run from the application context, with its stage events kept on disk."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.services import dataset_for, input_plans_for, pipeline_deps, plan_for
from physverify.core.download import BYTES_PER_MB
from physverify.core.errors import UsageError
from physverify.core.output import Echo
from physverify.datasets._base import DatasetSource, select_clips
from physverify.extraction.cache import StageCache
from physverify.extraction.error_profile import require_current_profile
from physverify.extraction.events import (
    ConsoleObserver,
    EventRecorder,
    plan_providers,
    summary_lines,
    write_events,
)
from physverify.extraction.pipeline import ExtractionPipeline, ExtractionReport
from physverify.extraction.plan import ResolvedPlan
from physverify.extraction.weights import NetworkDownloader, fetch, hub_cache
from physverify.runtime.runs import new_run
from physverify.schema.clip import ClipManifest
from physverify.schema.stages import parse_stages
from physverify.synth.migrate import manifest_dataset, rerender_clip_ids

EVENTS_FILE = "extraction.jsonl"

Record = Callable[
    [EventRecorder, Callable[[], ExtractionReport], Callable[[ExtractionReport], ExtractionReport]],
    ExtractionReport,
]
"""Runs an extraction and writes its stage events: :func:`recorded_run` or :func:`recorded`."""


def recorded_run[T](
    run_dir: Path,
    events_file: str,
    recorder: EventRecorder,
    run: Callable[[], T],
    report_of: Callable[[T], ExtractionReport],
    *,
    echo: Echo,
) -> T:
    """Print the run directory, call ``run`` and write its stage events to ``events_file``, also
    when it aborts."""
    echo(f"run: {run_dir}")
    return recorded(run_dir / events_file, recorder, run, report_of)


def recorded[T](
    events: Path,
    recorder: EventRecorder,
    run: Callable[[], T],
    report_of: Callable[[T], ExtractionReport],
) -> T:
    """Call ``run`` and write its stage events to ``events``, also when it aborts."""
    try:
        result = run()
    except Exception as exc:
        write_events(events, recorder.aborted_report(exc))
        raise
    write_events(events, report_of(result))
    return result


def route_events_file(route: str) -> str:
    """The events file of the extraction of input route ``route``."""
    return f"extraction.route-{route}.jsonl"


def run_extraction(
    context: AppContext,
    dataset: str,
    clips: Sequence[str],
    gt: Sequence[str],
    *,
    fps: float | None,
    max_side: int | None,
    subset: str | None,
    strict_profile: bool,
    echo: Echo,
) -> tuple[ExtractionReport, ...]:
    """Run the extraction plan, then every input route's plan that differs from it, over the
    selected clips and print each one's per-stage summary; ``strict_profile`` refuses a missing or
    stale error profile of any of them instead of warning."""
    source = dataset_for(context, dataset)
    selected = select_clips(source.clips(subset), clips)
    gt_stages = parse_stages(gt)
    main, routes = input_plans_for(
        context, dataset=dataset, gt_stages=gt_stages, fps=fps, max_side=max_side
    )
    if strict_profile:
        profile = pipeline_deps(context, None).error_profile
        for each in (main, *routes.values()):
            require_current_profile(each, profile)
    run_dir = new_run(context.settings, context.invocation.argv).run_dir
    first: Record = partial(recorded_run, run_dir, EVENTS_FILE, echo=echo)
    reports = [_extract(context, main, source, selected, first, echo=echo)]
    for name, route in routes.items():
        echo(f"route {name}:")
        then: Record = partial(recorded, run_dir / route_events_file(name))
        reports.append(_extract(context, route, source, selected, then, echo=echo))
    return tuple(reports)


def _extract(
    context: AppContext,
    plan: ResolvedPlan,
    source: DatasetSource,
    selected: Sequence[ClipManifest],
    record: Record,
    *,
    echo: Echo,
) -> ExtractionReport:
    recorder = EventRecorder(ConsoleObserver(echo))
    deps = pipeline_deps(context, recorder)

    def run() -> ExtractionReport:
        return ExtractionPipeline(plan, deps).run(source, selected)

    report = record(recorder, run, _same)
    for line in summary_lines(report):
        echo(line)
    return report


def fetch_weights(context: AppContext, *, echo: Echo) -> None:
    """Download and verify the weights of every provider in the active profile's plan."""
    settings = context.settings
    plan = plan_for(context, dataset="")
    downloader = NetworkDownloader()
    echo(f"hub cache: {hub_cache(settings.paths)}")
    for provider in plan_providers(plan):
        for spec in provider.weights:
            path = fetch(spec, settings.paths, downloader)
            size = _size_mb(path)
            echo(
                f"{provider.stage.value}/{provider.name} {spec.kind.value}:{spec.ref}@"
                f"{spec.revision[:12]} verified {size:.1f} MB {path}"
            )


def purge_rerendered(
    manifest: Path,
    roots: Sequence[Path],
    *,
    default_root: Callable[[], Path],
    echo: Echo,
) -> None:
    """Remove every cached stage of the clips the migration ``manifest`` re-rendered, from each of
    ``roots`` (``default_root()`` when empty)."""
    if not manifest.is_file():
        raise UsageError(f"no migration manifest at {manifest}")
    dataset = manifest_dataset(manifest)
    clip_ids = rerender_clip_ids(manifest)
    for root in roots or [default_root()]:
        cache = StageCache(root)
        entries = sum(cache.purge_clip(dataset, clip_id) for clip_id in clip_ids)
        echo(f"purge {root}: {entries} stage entries of {len(clip_ids)} clips removed")


def _size_mb(path: Path) -> float:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / BYTES_PER_MB


def _same(report: ExtractionReport) -> ExtractionReport:
    return report
