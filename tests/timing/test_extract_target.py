"""The extraction target: warm and timed replica cells over a scratch cache, A5's own timings."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from synthetic_clip import SyntheticFixtureDataset

from physverify.cli.context import PLUGIN_KINDS
from physverify.config.models import Settings
from physverify.datasets._base import DatasetSource
from physverify.extraction._base import StageEvent, StageProvider
from physverify.extraction.pipeline import ClipFailure, ExtractionReport, PipelineDeps
from physverify.extraction.plan import ResolvedPlan, resolve_plan
from physverify.runtime.heavy import NullLock
from physverify.schema.clip import ClipManifest
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.timing._base import TimingContext
from physverify.timing.recorder import StageSpan
from physverify.timing.records import (
    EXTRACT_COLD_EXTRA_S,
    EXTRACT_PEAK_RSS_MB,
    EXTRACT_SPAN_S,
    EXTRACT_WALL_S,
    TimingError,
    check_measures,
)
from physverify.timing.replica import TIMED, WARM, replica_id
from physverify.timing.report import SCRATCH_DIR, timing_dir
from physverify.timing.services import ClipSelection, TimingServices
from physverify.timing.targets.extract_pipeline import (
    ExtractPipeline,
    cold_extra,
    stage_measures,
)

FIXTURE = SyntheticFixtureDataset.name


@dataclass
class Journal:
    starts: list[StageEvent] = field(default_factory=list)
    ends: list[StageEvent] = field(default_factory=list)

    def on_stage_start(self, event: StageEvent) -> None:
        self.starts.append(event)

    def on_stage_end(self, event: StageEvent) -> None:
        self.ends.append(event)


@dataclass
class ScriptedPipeline:
    """Emits one uncached cell per clip and stage; a warm replica spends ``load_s`` more."""

    deps: PipelineDeps
    clock: Any
    stages: Mapping[Stage, float]
    load_s: float = 0.0
    failures: tuple[ClipFailure, ...] = ()
    runs: list[tuple[str, list[str]]] = field(default_factory=list)

    def run(self, dataset: DatasetSource, clips: Sequence[ClipManifest]) -> ExtractionReport:
        self.runs.append((dataset.name, [clip.clip_id for clip in clips]))
        observer = self.deps.observer
        assert observer is not None
        for clip in clips:
            for stage, seconds in self.stages.items():
                event = StageEvent(clip.clip_id, stage, "fake", cached=False, device="cpu")
                observer.on_stage_start(event)
                self.clock.advance(seconds + (self.load_s if clip.clip_id.endswith(WARM) else 0))
                observer.on_stage_end(replace(event, wall_s=seconds / 2, peak_rss_mb=seconds * 100))
        return ExtractionReport(events=(), failures=self.failures, keys={})


@dataclass
class ScriptedFactory:
    clock: Any
    stages: Mapping[Stage, float]
    load_s: float = 0.0
    failures: tuple[ClipFailure, ...] = ()
    built: list[ScriptedPipeline] = field(default_factory=list)

    def __call__(self, plan: ResolvedPlan, deps: PipelineDeps) -> ScriptedPipeline:
        pipeline = ScriptedPipeline(deps, self.clock, self.stages, self.load_s, self.failures)
        self.built.append(pipeline)
        return pipeline


def no_plan(settings: Settings, providers: Mapping[str, type[StageProvider]], dataset: str) -> Any:
    return None


def ground_truth_plan(
    settings: Settings, providers: Mapping[str, type[StageProvider]], dataset: str
) -> ResolvedPlan:
    return resolve_plan(settings, providers, dataset=dataset, gt_stages=STAGE_ORDER)


def context(settings: Settings, tmp_path: Path, observer: Journal | None = None) -> TimingContext:
    return TimingContext(settings, "cpu", 1, tmp_path / "run", observer)


def span(clip_id: str, stage: str, span_s: float, wall_s: float = 1.0) -> StageSpan:
    return StageSpan(clip_id, stage, "fake", "cpu", wall_s, 10.0, span_s, span_s, None)


@pytest.fixture
def scripted(fake_clock: Callable[..., Any]) -> ScriptedFactory:
    return ScriptedFactory(fake_clock(), {Stage.DETECT: 2.0, Stage.KINEMATICS: 0.5}, load_s=3.0)


def test_measures_come_from_the_timed_cell(
    make_settings: Callable[..., Settings],
    make_services: Callable[..., TimingServices],
    fake_memory: Callable[..., Any],
    clip_factory: Callable[..., ClipManifest],
    scripted: ScriptedFactory,
    tmp_path: Path,
) -> None:
    services = make_services(clock=scripted.clock, memory=lambda device: fake_memory(64.0))
    target = ExtractPipeline(services, pipeline_factory=scripted, plan_factory=no_plan)
    journal = Journal()
    measures = target.run(clip_factory("c0"), context(make_settings(), tmp_path, journal))
    check_measures(measures)
    assert scripted.built[0].runs == [
        ("fake_ds", [replica_id("c0", WARM), replica_id("c0", TIMED)])
    ]
    assert measures["extract.detect.wall_s"] == 1.0
    assert measures["extract.detect.peak_rss_mb"] == 200.0
    assert measures["extract.detect.span_s"] == 2.0
    assert measures["extract.detect.cuda_peak_mb"] == 64.0
    assert measures["extract.detect.cold_span_s"] == 5.0
    assert measures["extract.kinematics.cold_wall_s"] == 0.25
    assert measures[EXTRACT_WALL_S] == 1.25
    assert measures[EXTRACT_SPAN_S] == 2.5
    assert measures[EXTRACT_PEAK_RSS_MB] == 200.0
    assert measures[EXTRACT_COLD_EXTRA_S] == 6.0
    assert len(journal.ends) == 4


def test_each_call_uses_a_fresh_scratch_cache_that_is_removed(
    make_settings: Callable[..., Settings],
    make_services: Callable[..., TimingServices],
    clip_factory: Callable[..., ClipManifest],
    scripted: ScriptedFactory,
    tmp_path: Path,
) -> None:
    settings = make_settings()
    target = ExtractPipeline(
        make_services(clock=scripted.clock), pipeline_factory=scripted, plan_factory=no_plan
    )
    ctx = context(settings, tmp_path)
    target.run(clip_factory("c0"), ctx)
    target.run(clip_factory("c0"), ctx)
    roots = [pipeline.deps.cache.root for pipeline in scripted.built]
    assert roots == [
        timing_dir(ctx.run_dir) / SCRATCH_DIR / "extract-0",
        timing_dir(ctx.run_dir) / SCRATCH_DIR / "extract-1",
    ]
    assert not (timing_dir(ctx.run_dir) / SCRATCH_DIR).exists()
    assert roots[0] != settings.paths.cache_dir


def test_failed_cells_raise_and_still_remove_the_scratch(
    make_settings: Callable[..., Settings],
    make_services: Callable[..., TimingServices],
    clip_factory: Callable[..., ClipManifest],
    fake_clock: Callable[..., Any],
    tmp_path: Path,
) -> None:
    failure = ClipFailure(replica_id("c0", TIMED), Stage.DEPTH, "out of memory")
    factory = ScriptedFactory(fake_clock(), {Stage.DETECT: 1.0}, failures=(failure,))
    target = ExtractPipeline(make_services(), pipeline_factory=factory, plan_factory=no_plan)
    with pytest.raises(TimingError, match="depth: out of memory"):
        target.run(clip_factory("c0"), context(make_settings(), tmp_path))
    assert not factory.built[0].deps.cache.root.exists()


def test_the_lock_is_lazy_and_never_built_without_a_heavy_stage(
    make_settings: Callable[..., Settings],
    make_services: Callable[..., TimingServices],
    clip_factory: Callable[..., ClipManifest],
    scripted: ScriptedFactory,
    tmp_path: Path,
) -> None:
    built: list[str] = []

    def lock() -> NullLock:
        built.append("lock")
        return NullLock("timing test")

    services = make_services(clock=scripted.clock, lock_factory=lock)
    target = ExtractPipeline(services, pipeline_factory=scripted, plan_factory=no_plan)
    target.run(clip_factory("c0"), context(make_settings(), tmp_path))
    assert scripted.built[0].deps.lock.acquisitions == 0
    assert built == []


def test_real_pipeline_copies_a5_timings_bit_for_bit(
    make_settings: Callable[..., Settings],
    make_services: Callable[..., TimingServices],
    tmp_path: Path,
) -> None:
    settings = make_settings()
    services = make_services(
        datasets={FIXTURE: SyntheticFixtureDataset},
        providers=dict(PLUGIN_KINDS["providers"]()),
        selection=ClipSelection(FIXTURE),
        clock=time.perf_counter,
        cpu_clock=time.process_time,
    )
    clip = SyntheticFixtureDataset(settings).clips()[0]
    journal = Journal()
    target = ExtractPipeline(services, plan_factory=ground_truth_plan)
    ctx = context(settings, tmp_path, journal)
    for _ in range(2):
        journal.ends.clear()
        measures = target.run(clip, ctx)
        check_measures(measures)
        assert not any(event.cached for event in journal.ends)
        timed = [e for e in journal.ends if e.clip_id == replica_id(clip.clip_id, TIMED)]
        assert {event.stage for event in timed} == set(STAGE_ORDER)
        for event in timed:
            assert measures[f"extract.{event.stage.value}.wall_s"] == event.wall_s
            assert measures[f"extract.{event.stage.value}.peak_rss_mb"] == event.peak_rss_mb
        spans = [measures[f"extract.{stage.value}.span_s"] for stage in STAGE_ORDER]
        assert measures[EXTRACT_SPAN_S] == pytest.approx(sum(spans))
    assert not (timing_dir(ctx.run_dir) / SCRATCH_DIR).exists()
    assert not settings.paths.cache_dir.exists()


def test_describe_names_the_plan(
    make_settings: Callable[..., Settings],
    make_services: Callable[..., TimingServices],
    tmp_path: Path,
) -> None:
    services = make_services(providers=dict(PLUGIN_KINDS["providers"]()))
    settings = make_settings()
    info = ExtractPipeline(services).describe(context(settings, tmp_path))
    plan = resolve_plan(settings, services.providers, dataset="fake_ds")
    assert info["dataset"] == "fake_ds"
    assert info["extraction_profile"] == plan.profile
    assert {key: info[key] for key in info if key.startswith("stage.")} == {
        f"stage.{stage.value}": name for stage, name in plan.names.items()
    }


def test_no_timed_cell_is_an_error() -> None:
    with pytest.raises(TimingError, match="no uncached stage"):
        stage_measures("c0", [span(replica_id("c0", WARM), "detect", 1.0)])


def test_cold_extra_never_goes_negative() -> None:
    warm = {"detect": span("w", "detect", 1.0), "lift": span("w", "lift", 0.25)}
    timed = {"detect": span("t", "detect", 1.5), "lift": span("t", "lift", 0.25)}
    assert cold_extra(warm, timed) == 0.0
    assert cold_extra(timed, warm) == 0.5
