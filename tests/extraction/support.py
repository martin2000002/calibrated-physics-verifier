"""Builders shared by the extraction tests: contexts, plans, pipeline dependencies, keys."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from fakes import JOURNAL
from synthetic_clip import SyntheticClip, SyntheticFixtureDataset

from physverify.config.models import PathsSettings, Settings
from physverify.datasets._base import GroundTruthStore
from physverify.extraction._base import (
    GROUND_TRUTH,
    StageContext,
    StageEvent,
    StageProvider,
)
from physverify.extraction.cache import StageCache, stage_key
from physverify.extraction.pipeline import (
    ExtractionPipeline,
    ExtractionReport,
    LazyLock,
    PipelineDeps,
)
from physverify.extraction.plan import ResolvedPlan, resolve_plan
from physverify.extraction.providers.contacts_proximity import ProximityContacts
from physverify.extraction.providers.gt_camera import GroundTruthCamera
from physverify.extraction.providers.gt_depth import GroundTruthDepth
from physverify.extraction.providers.gt_detect import GroundTruthDetect
from physverify.extraction.providers.gt_lift import GroundTruthLift
from physverify.extraction.providers.gt_scale import GroundTruthScale
from physverify.extraction.providers.gt_track import GroundTruthTrack
from physverify.extraction.providers.kinematics_savgol import SavgolKinematics
from physverify.schema.clip import ClipManifest, FrameSampling, SamplingSpec, resolve_sampling
from physverify.schema.error_profile import ErrorEntry, ErrorProfile
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable

UNUSED_PATHS = PathsSettings.model_validate({"data_dir": "/nonexistent/physverify-tests"})


def sampling_of(clip: ClipManifest, spec: SamplingSpec | None = None) -> FrameSampling:
    return resolve_sampling(clip, spec or SamplingSpec())


def make_context(
    clip: SyntheticClip,
    *,
    upstream: Mapping[Stage, StageOutput] | None = None,
    plan: Mapping[Stage, str] | None = None,
    profile: ErrorProfile | None = None,
    gt: GroundTruthStore | None = None,
    manifest: ClipManifest | None = None,
) -> StageContext:
    manifest = manifest or clip.manifest
    return StageContext(
        clip=manifest,
        video=clip.video,
        sampling=sampling_of(manifest),
        upstream=dict(upstream or {}),
        plan=dict(plan or dict.fromkeys(STAGE_ORDER, GROUND_TRUTH)),
        gt=gt,
        error_profile=profile,
        device="cpu",
        threads=1,
        paths=UNUSED_PATHS,
    )


def run_provider(provider: type[StageProvider], ctx: StageContext, **params: Any) -> StageOutput:
    return provider(provider.Params(**params)).run(ctx)


def entry(stage: Stage, provider: str, quantity: str, sigma: float, **extra: Any) -> ErrorEntry:
    return ErrorEntry(
        stage=stage,
        provider=provider,
        provider_version="1",
        quantity=quantity,
        unit="",
        bias=extra.pop("bias", 0.0),
        sigma=sigma,
        n=extra.pop("n", 50),
        **extra,
    )


def profile_of(*entries: ErrorEntry) -> ErrorProfile:
    return ErrorProfile(
        created_utc="2026-09-24T00:00:00+00:00", run_id="test", commit="test", entries=entries
    )


def with_sigma(profile: ErrorProfile, key: tuple[Stage, str, str], sigma: float) -> ErrorProfile:
    """``profile`` with the sigma of one (stage, provider, quantity) entry replaced."""
    entries = tuple(
        replace(e, sigma=sigma) if (e.stage, e.provider, e.quantity) == key else e
        for e in profile.entries
    )
    return replace(profile, entries=entries)


class FakeRss:
    def reset(self) -> None:
        return None

    def peak_mb(self) -> float:
        return 1.0


class JournalObserver:
    """Logs every stage start into the fakes' journal, next to lock and run entries."""

    def on_stage_start(self, event: StageEvent) -> None:
        JOURNAL.calls.append(f"start:{event.stage.value}:{event.clip_id}")

    def on_stage_end(self, event: StageEvent) -> None:
        return None


class CountingLock:
    """Heavy-lock factory that counts calls and logs acquire/release into the journal."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> AbstractContextManager[object]:
        self.calls += 1
        return self._held()

    @contextmanager
    def _held(self) -> Iterator[None]:
        JOURNAL.calls.append("lock:acquire")
        try:
            yield
        finally:
            JOURNAL.calls.append("lock:release")


def make_deps(
    settings: Settings,
    *,
    lock: Callable[[], AbstractContextManager[object]] | None = None,
    profile: ErrorProfile | None = None,
) -> PipelineDeps:
    return PipelineDeps(
        cache=StageCache(settings.paths.cache_dir),
        lock=LazyLock(lock or nullcontext),
        paths=settings.paths,
        device="cpu",
        threads=1,
        error_profile=profile,
        observer=JournalObserver(),
        rss=FakeRss(),
    )


def run_pipeline(
    settings: Settings,
    providers: Mapping[str, type[StageProvider]],
    dataset: SyntheticFixtureDataset,
    *,
    gt_stages: tuple[Stage, ...] = (),
    lock: Callable[[], AbstractContextManager[object]] | None = None,
    profile: ErrorProfile | None = None,
    clips: Sequence[ClipManifest] | None = None,
) -> ExtractionReport:
    plan = resolve_plan(settings, providers, dataset=dataset.name, gt_stages=gt_stages)
    deps = make_deps(settings, lock=lock, profile=profile)
    return ExtractionPipeline(plan, deps).run(dataset, clips or dataset.clips())


def load_output(
    settings: Settings, report: ExtractionReport, plan: ResolvedPlan, clip_id: str, stage: Stage
) -> StageOutput:
    cache = StageCache(settings.paths.cache_dir)
    output = cache.load(
        "synthetic_fixture", clip_id, stage, plan.step(stage).name, report.keys[(clip_id, stage)]
    )
    assert output is not None
    return output


def chain_keys(
    plan: ResolvedPlan,
    clip: ClipManifest,
    *,
    gt_version: str | None,
    profile: ErrorProfile | None,
    device: str = "cpu",
) -> dict[Stage, str]:
    """Stage keys in plan order, computed the way the pipeline computes them."""
    sampling = resolve_sampling(clip, plan.sampling)
    keys: dict[Stage, str] = {}
    for step in plan.stages:
        keys[step.stage] = stage_key(
            step,
            clip_id=clip.clip_id,
            sampling=sampling,
            upstream=keys,
            plan=plan.names,
            gt_version=gt_version if step.provider.is_ground_truth else None,
            error_profile=profile,
            device=device,
        )
    return keys


TRUTH_CHAIN: tuple[type[StageProvider], ...] = (
    GroundTruthDetect,
    GroundTruthTrack,
    GroundTruthDepth,
    GroundTruthCamera,
    GroundTruthLift,
    GroundTruthScale,
    SavgolKinematics,
    ProximityContacts,
)
"""Ground truth down to SCALE, then the light perceived KINEMATICS and CONTACTS."""


def truth_outputs(clip: SyntheticClip) -> dict[Stage, StageOutput]:
    """Every stage output of the fixture from ground truth, with a LIFT ``source_key``."""
    outputs: dict[Stage, StageOutput] = {}
    for provider in TRUTH_CHAIN:
        ctx = make_context(clip, upstream=outputs, gt=clip.ground_truth)
        output = run_provider(provider, ctx)
        if isinstance(output, ObjectStateTable) and provider is GroundTruthLift:
            output = replace(output, source_key="k-lift")
        outputs[provider.stage] = output
    return outputs


def plan_overrides(**names: str) -> list[str]:
    """``--set extraction.plan.<stage>=<name>`` for each keyword."""
    return [f"extraction.plan.{stage}={name}" for stage, name in names.items()]


def files_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []
