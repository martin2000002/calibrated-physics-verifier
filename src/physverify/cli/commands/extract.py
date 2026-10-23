"""``physverify extract``: run the stage plan over a dataset, fetch its weights (ADR 0005, 3.12)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from physverify.cli.context import AppContext, app_context, heavy_command
from physverify.core.download import BYTES_PER_MB
from physverify.core.errors import EXIT_ERROR, UsageError
from physverify.datasets._base import DatasetSource
from physverify.extraction._base import StageEvent, StageObserver, StageProvider
from physverify.extraction.cache import StageCache
from physverify.extraction.pipeline import (
    ClipFailure,
    ExtractionPipeline,
    ExtractionReport,
    LazyLock,
    PipelineDeps,
    ProcRssProbe,
)
from physverify.extraction.plan import ResolvedPlan, resolve_plan
from physverify.extraction.weights import NetworkDownloader, fetch, hub_cache
from physverify.runtime.runs import new_run
from physverify.schema.clip import ClipManifest
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.stages import STAGE_ORDER, parse_stages
from physverify.synth.migrate import manifest_dataset, rerender_clip_ids

NAME = "extract"
EVENTS_FILE = "extraction.jsonl"

app = typer.Typer(
    help="Perception pipeline: from video to states and contacts.",
    no_args_is_help=True,
)


class ConsoleObserver:
    """Prints one line per finished stage and clip."""

    def __init__(self, echo: Callable[[str], None]) -> None:
        self._echo = echo

    def on_stage_start(self, event: StageEvent) -> None:
        return None

    def on_stage_end(self, event: StageEvent) -> None:
        status = "cached" if event.cached else f"{event.wall_s:.2f} s {_mb(event.peak_rss_mb)}"
        self._echo(f"{event.clip_id} {event.stage.value}/{event.provider}: {status}")


class EventRecorder:
    """Keeps finished events and the stage in progress, so an aborted run still leaves its log."""

    def __init__(self, inner: StageObserver) -> None:
        self._inner = inner
        self.finished: list[StageEvent] = []
        self.current: StageEvent | None = None

    def on_stage_start(self, event: StageEvent) -> None:
        self.current = event
        self._inner.on_stage_start(event)

    def on_stage_end(self, event: StageEvent) -> None:
        self.current = None
        self.finished.append(event)
        self._inner.on_stage_end(event)

    def aborted_report(self, error: BaseException) -> ExtractionReport:
        """Finished events plus the stage in progress, failed with the aborting error."""
        message = f"aborted: {type(error).__name__}: {error}"
        current = self.current
        failures = (
            () if current is None else (ClipFailure(current.clip_id, current.stage, message),)
        )
        return ExtractionReport(tuple(self.finished), failures, {})


@app.command("run", help="Run the extraction plan over a dataset.")
def run(
    dataset: Annotated[str, typer.Option(help="Dataset name.")],
    clips: Annotated[list[str] | None, typer.Option(help="Clip ids (repeatable).")] = None,
    gt: Annotated[list[str] | None, typer.Option(help="Stages replaced by ground truth.")] = None,
    fps: Annotated[float | None, typer.Option(help="Target sampling fps.")] = None,
    max_side: Annotated[int | None, typer.Option(help="Longest frame side in pixels.")] = None,
    subset: Annotated[str | None, typer.Option(help="Dataset subset.")] = None,
) -> None:
    context = app_context()
    settings = context.settings
    source = resolve_dataset(context, dataset)
    selected = select_clips(source.clips(subset), clips or ())
    plan = resolve_plan(
        settings,
        context.registry("providers"),
        dataset=dataset,
        gt_stages=parse_stages(gt or ()),
        fps=fps,
        max_side=max_side,
    )
    run_context = new_run(settings, context.invocation.argv)
    events_path = run_context.run_dir / EVENTS_FILE
    typer.echo(f"run: {run_context.run_dir}")
    recorder = EventRecorder(ConsoleObserver(typer.echo))
    try:
        report = ExtractionPipeline(plan, pipeline_deps(context, recorder)).run(source, selected)
    except Exception as exc:
        write_events(events_path, recorder.aborted_report(exc))
        raise
    write_events(events_path, report)
    for line in summary_lines(report):
        typer.echo(line)
    if report.failures:
        raise typer.Exit(code=EXIT_ERROR)


@app.command("fetch-weights", help="Download the weights of the active profile's providers.")
@heavy_command
def fetch_weights() -> None:
    context = app_context()
    settings = context.settings
    plan = resolve_plan(settings, context.registry("providers"), dataset="")
    downloader = NetworkDownloader()
    typer.echo(f"hub cache: {hub_cache(settings.paths)}")
    for provider in plan_providers(plan):
        for spec in provider.weights:
            path = fetch(spec, settings.paths, downloader)
            size = _size_mb(path)
            typer.echo(
                f"{provider.stage.value}/{provider.name} {spec.kind.value}:{spec.ref}@"
                f"{spec.revision[:12]} verified {size:.1f} MB {path}"
            )


@app.command(
    "purge",
    help="Remove every cached stage of the clips a dataset migration re-rendered (all providers).",
)
def purge(
    clips_from: Annotated[
        Path, typer.Option(help="Migration manifest (<dataset>/migrations/<name>.json).")
    ],
    cache_root: Annotated[
        list[Path] | None,
        typer.Option(help="Cache root (repeatable; default: paths.cache_dir)."),
    ] = None,
) -> None:
    if not clips_from.is_file():
        raise UsageError(f"no migration manifest at {clips_from}")
    dataset = manifest_dataset(clips_from)
    clip_ids = rerender_clip_ids(clips_from)
    roots = cache_root or [app_context().settings.paths.cache_dir]
    for root in roots:
        cache = StageCache(root)
        entries = sum(cache.purge_clip(dataset, clip_id) for clip_id in clip_ids)
        typer.echo(f"purge {root}: {entries} stage entries of {len(clip_ids)} clips removed")


def resolve_dataset(context: AppContext, name: str) -> DatasetSource:
    registry = context.registry("datasets")
    if name not in registry:
        raise UsageError(f"unknown dataset {name!r}; known: {sorted(registry)}")
    source: DatasetSource = registry[name](context.settings)
    return source


def select_clips(clips: Sequence[ClipManifest], wanted: Sequence[str]) -> list[ClipManifest]:
    if not wanted:
        return list(clips)
    by_id = {clip.clip_id: clip for clip in clips}
    unknown = sorted(set(wanted) - set(by_id))
    if unknown:
        raise UsageError(f"unknown clip ids {unknown}")
    return [by_id[clip_id] for clip_id in dict.fromkeys(wanted)]


def pipeline_deps(
    context: AppContext,
    observer: StageObserver | None,
    heavy_lock: Callable[[], AbstractContextManager[object]] | None = None,
) -> PipelineDeps:
    """``heavy_lock`` replaces the machine lock, e.g. with one that refuses in commands that must
    never run a heavy stage."""
    settings = context.settings
    return PipelineDeps(
        cache=StageCache(settings.paths.cache_dir),
        lock=LazyLock(heavy_lock or context.heavy_lock),
        paths=settings.paths,
        device=settings.runtime.device,
        threads=settings.runtime.threads,
        error_profile=load_error_profile(settings.paths.error_profile),
        observer=observer,
        rss=ProcRssProbe(),
    )


def load_error_profile(path: Path) -> ErrorProfile | None:
    return ErrorProfile.load_json(path) if path.is_file() else None


def plan_providers(plan: ResolvedPlan) -> list[type[StageProvider]]:
    return list(dict.fromkeys(step.provider for step in plan.stages))


def event_record(event: StageEvent) -> dict[str, Any]:
    record = asdict(event)
    record["stage"] = event.stage.value
    return {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in record.items()}


def write_events(path: Path, report: ExtractionReport) -> None:
    lines = [json.dumps(event_record(event)) for event in report.events]
    lines += [
        json.dumps({"clip_id": f.clip_id, "stage": f.stage.value, "error": f.error})
        for f in report.failures
    ]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def summary_lines(report: ExtractionReport) -> list[str]:
    lines: list[str] = []
    for stage in STAGE_ORDER:
        events = [e for e in report.events if e.stage is stage]
        if not events:
            continue
        computed = [e for e in events if not e.cached]
        wall = sum(e.wall_s for e in computed)
        peak = max((e.peak_rss_mb for e in computed), default=math.nan)
        lines.append(
            f"{stage.value}/{events[0].provider}: cached {len(events) - len(computed)} "
            f"run {len(computed)} total {wall:.2f} s peak {_mb(peak)}"
        )
    lines += [f"FAILED {f.clip_id} at {f.stage.value}: {f.error}" for f in report.failures]
    return lines


def _mb(value: float) -> str:
    return "rss n/a" if math.isnan(value) else f"rss {value:.0f} MB"


def _size_mb(path: Path) -> float:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / BYTES_PER_MB
