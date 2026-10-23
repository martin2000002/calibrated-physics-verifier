"""Per-clip cost of the check suite: loading the perceived inputs, the checks and the policy.

The check suite is imported only when the target builds it, so building the registry does not
load the runner.
``verify.wall_s`` is one span over the input load and the runner (checks plus policy);
``verify.load_s`` and the per-check ``verify.<check_id>.wall_s`` are its components, the latter
timed again right after the span on the same in-memory inputs.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Final, Protocol

from physverify.checks._base import Check, CheckInputs, missing_inputs
from physverify.config.models import Settings
from physverify.core.errors import TimingError
from physverify.datasets._base import DatasetSource, dataset_from
from physverify.extraction._base import StageObserver
from physverify.extraction.pipeline import ExtractionPipeline, PipelineDeps
from physverify.extraction.plan import ResolvedPlan
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest
from physverify.timing._base import TimingContext
from physverify.timing.hooks import TargetHooks
from physverify.timing.records import (
    VERIFY_CHECKS_S,
    VERIFY_CPU_S,
    VERIFY_LOAD_S,
    VERIFY_WALL_S,
    measure_name,
)
from physverify.timing.target_services import TimingServices
from physverify.timing.targets._extract_pipeline import (
    PipelineFactory,
    PlanFactory,
    perception_plan,
    pipeline_deps,
)
from physverify.verification._base import VerdictPolicy
from physverify.verification.inputs import (
    ClipInputs,
    InputSource,
    PerceivedSource,
    PipelineOutputLoader,
)

PREFIX: Final = "verify"
_NOT_SLUG: Final = re.compile(r"[^a-z0-9_]")


class Runner(Protocol):
    def run(self, source: InputSource, clips: Sequence[ClipManifest]) -> object: ...


SourceFactory = Callable[[DatasetSource, ResolvedPlan, PipelineDeps], InputSource]


@dataclass(frozen=True)
class Verifier:
    """What the check suite builds once per run: the checks, their runner, the perceived source."""

    checks: tuple[Check, ...]
    runner: Runner
    source: SourceFactory


SuiteFactory = Callable[
    [Settings, Mapping[str, type[Check]], Mapping[str, type[VerdictPolicy]]], Verifier
]


def load_suite(
    settings: Settings,
    checks: Mapping[str, type[Check]],
    policies: Mapping[str, type[VerdictPolicy]],
) -> Verifier:
    from physverify.verification.runner import configured_suite

    suite = configured_suite(settings, checks, policies)

    def perceived(dataset: DatasetSource, plan: ResolvedPlan, deps: PipelineDeps) -> InputSource:
        loader = PipelineOutputLoader(plan, deps)
        return PerceivedSource(dataset, loader, InputLevel.PERCEIVED)

    return Verifier(suite.checks, suite.runner(), perceived)


def check_measure(check_id: str) -> str:
    """``R4_jump`` -> ``verify.r4_jump.wall_s`` (measure names are lower-case slugs)."""
    return measure_name(PREFIX, _NOT_SLUG.sub("_", check_id.lower()), "wall_s")


@dataclass(frozen=True)
class PreloadedSource:
    """A check-suite ``InputSource`` over in-memory inputs, so the runner times no loading."""

    loaded: tuple[ClipInputs, ...]

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        wanted = {clip.clip_id for clip in clips}
        return (item for item in self.loaded if item.clip.clip_id in wanted)


def only_inputs(clip: ClipManifest, loaded: Sequence[ClipInputs]) -> CheckInputs:
    for item in loaded:
        if item.error:
            raise TimingError(f"{clip.clip_id}: verification inputs failed: {item.error}")
    found = [item.inputs for item in loaded if item.inputs is not None]
    if len(found) != 1:
        raise TimingError(f"{clip.clip_id}: expected one set of inputs, got {len(found)}")
    return found[0]


class VerifyChecks(TargetHooks):
    name: ClassVar[str] = "verify_checks"
    requires_gpu: ClassVar[bool] = False

    def __init__(
        self,
        services: TimingServices,
        *,
        suite_factory: SuiteFactory = load_suite,
        pipeline_factory: PipelineFactory = ExtractionPipeline,
        plan_factory: PlanFactory = perception_plan,
    ) -> None:
        self._services = services
        self._suite_factory = suite_factory
        self._pipeline_factory = pipeline_factory
        self._plan_factory = plan_factory
        self._verifier: Verifier | None = None
        self._check_ids: tuple[str, ...] = ()
        self._prepare_failures: tuple[str, ...] = ()

    def prepare(self, clips: Sequence[ClipManifest], ctx: TimingContext) -> Mapping[str, float]:
        """Fills the persistent extraction cache (untimed; cached cells are not recomputed) and
        builds the checks, the policy and the thresholds once."""
        self._prepare_failures = tuple(self._fill_cache(clips, ctx))
        self._verifier = self._build(ctx)
        return {}

    def describe(self, ctx: TimingContext) -> Mapping[str, str]:
        info = {"input_level": InputLevel.PERCEIVED.value}
        if self._check_ids:
            info["checks"] = ",".join(self._check_ids)
        if self._prepare_failures:
            info["prepare_failures"] = "; ".join(self._prepare_failures)
        return info

    def close(self) -> None:
        self._verifier = None

    def run(self, clip: ClipManifest, ctx: TimingContext) -> Mapping[str, float]:
        verifier = self._verifier = self._verifier or self._build(ctx)
        source = self._input_source(verifier, clip, ctx)
        clock, cpu_clock = self._services.clock, self._services.cpu_clock
        started, cpu_started = clock(), cpu_clock()
        loaded = tuple(source.load([clip]))
        loaded_at = clock()
        inputs = only_inputs(clip, loaded)
        verifier.runner.run(PreloadedSource(loaded), [clip])
        ended, cpu_ended = clock(), cpu_clock()
        per_check = self._time_checks(verifier.checks, inputs)
        return {
            VERIFY_WALL_S: ended - started,
            VERIFY_LOAD_S: loaded_at - started,
            VERIFY_CPU_S: cpu_ended - cpu_started,
            VERIFY_CHECKS_S: sum(per_check.values()),
            **per_check,
        }

    def _build(self, ctx: TimingContext) -> Verifier:
        services = self._services
        verifier = self._suite_factory(ctx.settings, services.checks, services.policies)
        self._check_ids = tuple(type(check).check_id for check in verifier.checks)
        return verifier

    def _input_source(
        self, verifier: Verifier, clip: ClipManifest, ctx: TimingContext
    ) -> InputSource:
        dataset = dataset_from(self._services.datasets, clip.dataset, ctx.settings)
        plan = self._plan_factory(ctx.settings, self._services, clip.dataset)
        deps = self._deps(ctx, None)
        return verifier.source(dataset, plan, deps)

    def _deps(self, ctx: TimingContext, observer: StageObserver | None) -> PipelineDeps:
        """Over the persistent cache that ``extract run`` also uses."""
        return pipeline_deps(self._services, ctx, ctx.settings.paths.cache_dir, observer=observer)

    def _fill_cache(self, clips: Sequence[ClipManifest], ctx: TimingContext) -> list[str]:
        failures: list[str] = []
        for dataset_name in dict.fromkeys(clip.dataset for clip in clips):
            group = [clip for clip in clips if clip.dataset == dataset_name]
            plan = self._plan_factory(ctx.settings, self._services, dataset_name)
            dataset = dataset_from(self._services.datasets, dataset_name, ctx.settings)
            pipeline = self._pipeline_factory(plan, self._deps(ctx, ctx.observer))
            report = pipeline.run(dataset, group)
            failures += [f"{f.clip_id} {f.stage.value}: {f.error}" for f in report.failures]
        return failures

    def _time_checks(self, checks: Sequence[Check], inputs: CheckInputs) -> dict[str, float]:
        """Mirrors the runner: a check with missing inputs is skipped, and a check that raises still
        counts the time it took (the runner records it as an error and goes on)."""
        clock = self._services.clock
        spent: dict[str, float] = {}
        for check in checks:
            if missing_inputs(check, inputs):
                continue
            started = clock()
            with contextlib.suppress(Exception):
                check.evaluate(inputs)
            spent[check_measure(type(check).check_id)] = clock() - started
        return spent


PLUGINS = (VerifyChecks,)
