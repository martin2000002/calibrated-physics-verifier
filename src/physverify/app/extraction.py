"""An extraction run from the application context, with its stage events kept on disk."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.services import dataset_for, pipeline_deps, plan_for
from physverify.core.download import BYTES_PER_MB
from physverify.core.errors import UsageError
from physverify.core.output import Echo
from physverify.datasets._base import select_clips
from physverify.extraction.cache import StageCache
from physverify.extraction.events import (
    ConsoleObserver,
    EventRecorder,
    plan_providers,
    summary_lines,
    write_events,
)
from physverify.extraction.pipeline import ExtractionPipeline, ExtractionReport
from physverify.extraction.weights import NetworkDownloader, fetch, hub_cache
from physverify.runtime.runs import new_run
from physverify.schema.stages import parse_stages
from physverify.synth.migrate import manifest_dataset, rerender_clip_ids

EVENTS_FILE = "extraction.jsonl"


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

    report = recorded_run(run_dir, EVENTS_FILE, recorder, run, _same, echo=echo)
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
