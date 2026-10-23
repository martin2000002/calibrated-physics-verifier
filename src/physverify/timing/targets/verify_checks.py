"""Per-clip cost of A7's verification: loading the perceived inputs, the checks and the policy.

A7 is reached only through ``importlib`` (ADR 0006, section 3.7-3.8), so building the registry or
type-checking this module never needs A7's code; without its runner the target is skipped.
``verify.wall_s`` is one span over the input load and the runner (checks plus policy);
``verify.load_s`` and the per-check ``verify.<check_id>.wall_s`` are its components, the latter
timed again right after the span on the same in-memory inputs.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Protocol

from physverify.checks._base import Check, CheckInputs
from physverify.config.models import Settings
from physverify.datasets._base import DatasetSource
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
    TimingError,
    measure_name,
)
from physverify.timing.target_services import TimingServices, dataset_source
from physverify.timing.targets.extract_pipeline import (
    PipelineFactory,
    PlanFactory,
    perception_plan,
    pipeline_deps,
)
from physverify.verification._base import VerdictPolicy

PREFIX: Final = "verify"
RUNNER_MODULE: Final = "physverify.verification.runner"
CHECK_SET_MODULE: Final = "physverify.verification.check_set"
INPUTS_MODULE: Final = "physverify.verification.inputs"
_NOT_SLUG: Final = re.compile(r"[^a-z0-9_]")


class LoadedInputs(Protocol):
    """A7's ``ClipInputs``."""

    @property
    def clip(self) -> ClipManifest: ...

    @property
    def inputs(self) -> CheckInputs | None: ...

    @property
    def error(self) -> str | None: ...


class InputSource(Protocol):
    def load(self, clips: Sequence[ClipManifest]) -> Iterable[LoadedInputs]: ...


class Runner(Protocol):
    def run(self, source: Any, clips: Sequence[ClipManifest]) -> object: ...


SourceFactory = Callable[[DatasetSource, ResolvedPlan, PipelineDeps], InputSource]


@dataclass(frozen=True)
class Verifier:
    """What A7 builds once per run: the checks, the runner over them, and the perceived source."""

    checks: tuple[Check, ...]
    runner: Runner
    source: SourceFactory


A7Factory = Callable[
    [Settings, Mapping[str, type[Check]], Mapping[str, type[VerdictPolicy]]], Verifier
]


def load_a7(
    settings: Settings,
    checks: Mapping[str, type[Check]],
    policies: Mapping[str, type[VerdictPolicy]],
) -> Verifier:
    runner_api: Any = importlib.import_module(RUNNER_MODULE)
    check_set_api: Any = importlib.import_module(CHECK_SET_MODULE)
    inputs_api: Any = importlib.import_module(INPUTS_MODULE)
    config = settings.section(runner_api.VERIFICATION_SECTION, runner_api.VerificationSettings)
    overrides = settings.section(check_set_api.CHECKS_SECTION, check_set_api.ChecksSettings)
    built = tuple(check_set_api.build_checks(checks, overrides))
    policy = runner_api.build_policy(policies, config)
    thresholds = runner_api.load_thresholds(settings.paths.thresholds, config.thresholds)

    def perceived(dataset: DatasetSource, plan: ResolvedPlan, deps: PipelineDeps) -> InputSource:
        loader = inputs_api.PipelineOutputLoader(plan, deps)
        source: InputSource = inputs_api.PerceivedSource(dataset, loader, InputLevel.PERCEIVED)
        return source

    return Verifier(built, runner_api.VerificationRunner(built, policy, thresholds), perceived)


def check_measure(check_id: str) -> str:
    """``R4_jump`` -> ``verify.r4_jump.wall_s`` (measure names are lower-case slugs)."""
    return measure_name(PREFIX, _NOT_SLUG.sub("_", check_id.lower()), "wall_s")


def a7_missing(find_spec: Callable[[str], object | None]) -> str | None:
    missing = [name for name in (RUNNER_MODULE, INPUTS_MODULE) if find_spec(name) is None]
    return f"A7 verification runner not installed ({', '.join(missing)})" if missing else None


@dataclass(frozen=True)
class PreloadedSource:
    """An A7 ``InputSource`` over inputs already in memory, so the runner times no loading."""

    loaded: tuple[LoadedInputs, ...]

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[LoadedInputs]:
        wanted = {clip.clip_id for clip in clips}
        return (item for item in self.loaded if item.clip.clip_id in wanted)


def only_inputs(clip: ClipManifest, loaded: Sequence[LoadedInputs]) -> CheckInputs:
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
        a7_factory: A7Factory = load_a7,
        find_spec: Callable[[str], object | None] = importlib.util.find_spec,
        pipeline_factory: PipelineFactory = ExtractionPipeline,
        plan_factory: PlanFactory = perception_plan,
    ) -> None:
        self._services = services
        self._a7_factory = a7_factory
        self._find_spec = find_spec
        self._pipeline_factory = pipeline_factory
        self._plan_factory = plan_factory
        self._verifier: Verifier | None = None
        self._check_ids: tuple[str, ...] = ()
        self._prepare_failures: tuple[str, ...] = ()

    def unavailable(self, ctx: TimingContext) -> str | None:
        return a7_missing(self._find_spec)

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
        source = self._source(verifier, clip, ctx)
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
        verifier = self._a7_factory(ctx.settings, services.checks, services.policies)
        self._check_ids = tuple(type(check).check_id for check in verifier.checks)
        return verifier

    def _source(self, verifier: Verifier, clip: ClipManifest, ctx: TimingContext) -> InputSource:
        dataset = dataset_source(self._services, ctx.settings, clip.dataset)
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
            dataset = dataset_source(self._services, ctx.settings, dataset_name)
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
            if any(getattr(inputs, name) is None for name in check.requires):
                continue
            started = clock()
            with contextlib.suppress(Exception):
                check.evaluate(inputs)
            spent[check_measure(type(check).check_id)] = clock() - started
        return spent


PLUGINS = (VerifyChecks,)
