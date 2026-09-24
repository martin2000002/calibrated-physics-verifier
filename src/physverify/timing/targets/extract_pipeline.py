"""Per-clip cost of A5's extraction pipeline, read from its own stage events (no re-timing).

Each call runs the pipeline on a ``warm`` and a ``timed`` replica of the clip over a fresh scratch
cache. Per stage, the timed cell gives the steady-state cost: ``wall_s`` and ``peak_rss_mb`` exactly
as A5 reports them (provider compute only) plus the recorder's ``span_s`` (upstream cache loads,
compute and cache store), ``cpu_s`` and ``cuda_peak_mb``. The warm cell adds the model load
(``cold_*``). ``extract.span_s`` is the per-clip extraction total.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import count
from pathlib import Path
from typing import ClassVar, Final, Protocol

from physverify.cli.commands.extract import load_error_profile
from physverify.config.models import Settings
from physverify.datasets._base import DatasetSource
from physverify.extraction._base import StageObserver, StageProvider
from physverify.extraction.cache import StageCache
from physverify.extraction.pipeline import (
    ExtractionPipeline,
    ExtractionReport,
    LazyLock,
    PipelineDeps,
)
from physverify.extraction.plan import ResolvedPlan, resolve_plan
from physverify.schema.clip import ClipManifest
from physverify.timing._base import TimingContext
from physverify.timing.hooks import TargetHooks
from physverify.timing.recorder import StageSpan, TeeObserver
from physverify.timing.records import (
    EXTRACT_COLD_EXTRA_S,
    EXTRACT_PEAK_RSS_MB,
    EXTRACT_SPAN_S,
    EXTRACT_WALL_S,
    TimingError,
    measure_name,
)
from physverify.timing.replica import TIMED, WARM, replica_id, replica_source, replicas
from physverify.timing.report import discard_scratch, scratch_dir
from physverify.timing.services import TimingServices, dataset_source

PREFIX: Final = "extract"


class Pipeline(Protocol):
    def run(self, dataset: DatasetSource, clips: Sequence[ClipManifest]) -> ExtractionReport: ...


PipelineFactory = Callable[[ResolvedPlan, PipelineDeps], Pipeline]
PlanFactory = Callable[[Settings, Mapping[str, type[StageProvider]], str], ResolvedPlan]


def perception_plan(
    settings: Settings, providers: Mapping[str, type[StageProvider]], dataset: str
) -> ResolvedPlan:
    """The active extraction profile, every stage perceived (no ground-truth substitution)."""
    return resolve_plan(settings, providers, dataset=dataset)


def pipeline_deps(
    services: TimingServices,
    ctx: TimingContext,
    cache_root: Path,
    *,
    observer: StageObserver | None,
) -> PipelineDeps:
    """Deps of a pipeline run inside ``timing run``: the lazy lock never builds a ``HeavyLock``."""
    paths = ctx.settings.paths
    return PipelineDeps(
        cache=StageCache(cache_root),
        lock=LazyLock(services.lock_factory),
        paths=paths,
        device=ctx.device,
        threads=ctx.threads,
        error_profile=load_error_profile(paths.error_profile),
        observer=observer,
        rss=services.rss(),
    )


def check_report(report: ExtractionReport) -> None:
    if report.failures:
        details = "; ".join(f"{f.clip_id} {f.stage.value}: {f.error}" for f in report.failures)
        raise TimingError(f"extraction failed: {details}")


class ExtractPipeline(TargetHooks):
    name: ClassVar[str] = "extract_pipeline"
    requires_gpu: ClassVar[bool] = False

    def __init__(
        self,
        services: TimingServices,
        *,
        pipeline_factory: PipelineFactory = ExtractionPipeline,
        plan_factory: PlanFactory = perception_plan,
    ) -> None:
        self._services = services
        self._pipeline_factory = pipeline_factory
        self._plan_factory = plan_factory
        self._calls = count()

    def describe(self, ctx: TimingContext) -> Mapping[str, str]:
        dataset = self._services.selection.dataset or ""
        plan = self._plan_factory(ctx.settings, self._services.providers, dataset)
        stages = {f"stage.{stage.value}": provider for stage, provider in plan.names.items()}
        return {"dataset": dataset, "extraction_profile": plan.profile, **stages}

    def run(self, clip: ClipManifest, ctx: TimingContext) -> Mapping[str, float]:
        scratch = scratch_dir(ctx.run_dir, PREFIX, next(self._calls))
        try:
            spans = self._extract(clip, ctx, scratch)
        finally:
            discard_scratch(scratch)
        return stage_measures(clip.clip_id, spans)

    def _extract(
        self, clip: ClipManifest, ctx: TimingContext, scratch: Path
    ) -> tuple[StageSpan, ...]:
        plan = self._plan_factory(ctx.settings, self._services.providers, clip.dataset)
        recorder = self._services.recorder(ctx.device)
        deps = pipeline_deps(
            self._services, ctx, scratch, observer=TeeObserver(recorder, ctx.observer)
        )
        copies = replicas(clip, (WARM, TIMED))
        source = dataset_source(self._services, ctx.settings, clip.dataset)
        report = self._pipeline_factory(plan, deps).run(
            replica_source(source, dict.fromkeys(copies, clip)), list(copies.values())
        )
        check_report(report)
        return recorder.spans


def _cell(spans: Sequence[StageSpan], clip_id: str, tag: str) -> dict[str, StageSpan]:
    wanted = replica_id(clip_id, tag)
    return {span.stage: span for span in spans if span.clip_id == wanted}


def stage_measures(clip_id: str, spans: Sequence[StageSpan]) -> dict[str, float]:
    """Per-stage and total measures of one call; ``wall_s``/``peak_rss_mb`` are A5's values."""
    timed, warm = _cell(spans, clip_id, TIMED), _cell(spans, clip_id, WARM)
    if not timed:
        raise TimingError(f"{clip_id}: no uncached stage ran in the timed cell")
    measures: dict[str, float] = {}
    for stage, span in timed.items():
        measures[measure_name(PREFIX, stage, "wall_s")] = span.wall_s
        measures[measure_name(PREFIX, stage, "peak_rss_mb")] = span.peak_rss_mb
        measures[measure_name(PREFIX, stage, "span_s")] = span.span_s
        measures[measure_name(PREFIX, stage, "cpu_s")] = span.cpu_s
        if span.cuda_peak_mb is not None:
            measures[measure_name(PREFIX, stage, "cuda_peak_mb")] = span.cuda_peak_mb
    for stage, span in warm.items():
        measures[measure_name(PREFIX, stage, "cold_wall_s")] = span.wall_s
        measures[measure_name(PREFIX, stage, "cold_span_s")] = span.span_s
    measures[EXTRACT_WALL_S] = sum(span.wall_s for span in timed.values())
    measures[EXTRACT_SPAN_S] = sum(span.span_s for span in timed.values())
    measures[EXTRACT_PEAK_RSS_MB] = max(span.peak_rss_mb for span in timed.values())
    measures[EXTRACT_COLD_EXTRA_S] = cold_extra(warm, timed)
    return measures


def cold_extra(warm: Mapping[str, StageSpan], timed: Mapping[str, StageSpan]) -> float:
    """Extra span of the warm cell over the timed one (model loads). Light stages can come out a
    little faster in the warm cell; a negative total is that noise, so it is reported as 0."""
    extra = sum(warm[stage].span_s - timed[stage].span_s for stage in timed.keys() & warm.keys())
    return max(extra, 0.0)


PLUGINS = (ExtractPipeline,)
