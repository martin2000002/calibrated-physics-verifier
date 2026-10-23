"""The verification target: A7 via importlib, one span over load plus runner, per-check times."""

from __future__ import annotations

import subprocess
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from physverify.app import plugins
from physverify.config.models import Settings
from physverify.datasets._base import DatasetSource
from physverify.extraction.pipeline import ClipFailure, ExtractionReport, PipelineDeps
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest
from physverify.schema.stages import Stage
from physverify.timing._base import TimingContext
from physverify.timing.records import (
    VERIFY_CHECKS_S,
    VERIFY_CPU_S,
    VERIFY_LOAD_S,
    VERIFY_WALL_S,
    TimingError,
    check_measures,
)
from physverify.timing.services import TimingServices
from physverify.timing.targets.verify_checks import (
    INPUTS_MODULE,
    RUNNER_MODULE,
    Verifier,
    VerifyChecks,
    check_measure,
    load_a7,
)

PLAN: Any = object()
INPUTS: Any = object()
CHECK: Any = object()


@dataclass(frozen=True)
class Loaded:
    clip: ClipManifest
    inputs: Any
    error: str | None = None


def fake_check(
    check_id: str,
    seconds: float,
    clock: Any,
    *,
    requires: frozenset[str] = frozenset(),
    fails: bool = False,
    inputs: Any = INPUTS,
) -> Any:
    def evaluate(self: Any, given: Any) -> None:
        assert given is inputs
        clock.advance(seconds)
        if fails:
            raise ValueError("contacts is None")

    attrs = {"check_id": check_id, "requires": requires, "evaluate": evaluate}
    return type(f"Check_{check_id}", (), attrs)()


@dataclass
class FakeRunner:
    clock: Any
    cpu_clock: Any
    seconds: float
    seen: list[list[str]] = field(default_factory=list)

    def run(self, source: Any, clips: Sequence[ClipManifest]) -> None:
        self.seen.append([item.clip.clip_id for item in source.load(clips)])
        self.clock.advance(self.seconds)
        self.cpu_clock.advance(self.seconds * 2)


@dataclass
class FakeSource:
    clock: Any
    seconds: float
    error: str | None = None
    loads: int = 0

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[Loaded]:
        self.loads += 1
        self.clock.advance(self.seconds)
        return iter([Loaded(clip, None if self.error else INPUTS, self.error) for clip in clips])


@dataclass
class FakeA7:
    """An ``A7Factory``: two checks (0.125 s and 0.0625 s), a 0.5 s runner, a 0.25 s load."""

    clock: Any
    cpu_clock: Any
    error: str | None = None
    builds: int = 0
    sources: list[tuple[str, Any, PipelineDeps]] = field(default_factory=list)
    runner: FakeRunner = field(init=False)
    source: FakeSource = field(init=False)

    def __post_init__(self) -> None:
        self.runner = FakeRunner(self.clock, self.cpu_clock, 0.5)
        self.source = FakeSource(self.clock, 0.25, self.error)

    def __call__(self, settings: Settings, checks: Any, policies: Any) -> Verifier:
        self.builds += 1
        built = (fake_check("R4_jump", 0.125, self.clock), fake_check("R1-pen", 0.0625, self.clock))
        return Verifier(built, self.runner, self.make_source)

    def make_source(self, dataset: DatasetSource, plan: Any, deps: PipelineDeps) -> FakeSource:
        self.sources.append((dataset.name, plan, deps))
        return self.source


@dataclass
class FakeFill:
    failures: tuple[ClipFailure, ...] = ()
    runs: list[tuple[str, list[str], PipelineDeps]] = field(default_factory=list)

    def __call__(self, plan: Any, deps: PipelineDeps) -> FakeFill:
        self._deps = deps
        return self

    def run(self, dataset: DatasetSource, clips: Sequence[ClipManifest]) -> ExtractionReport:
        ids = [clip.clip_id for clip in clips]
        self.runs.append((dataset.name, ids, self._deps))
        failures = tuple(failure for failure in self.failures if failure.clip_id in ids)
        return ExtractionReport(events=(), failures=failures, keys={})


def plan_of(settings: Settings, providers: Any, dataset: str) -> Any:
    return PLAN


def found(name: str) -> object:
    return object()


@dataclass
class Setup:
    services: TimingServices
    a7: FakeA7
    fill: FakeFill
    ctx: TimingContext

    def target(self, **changes: Any) -> VerifyChecks:
        options: dict[str, Any] = {
            "a7_factory": self.a7,
            "find_spec": found,
            "pipeline_factory": self.fill,
            "plan_factory": plan_of,
        }
        return VerifyChecks(self.services, **{**options, **changes})


@pytest.fixture
def setup(
    make_settings: Callable[..., Settings],
    make_services: Callable[..., TimingServices],
    fake_clock: Callable[..., Any],
    tmp_path: Path,
) -> Setup:
    clock, cpu_clock = fake_clock(), fake_clock()
    services = make_services(clock=clock, cpu_clock=cpu_clock)
    observer: Any = object()
    ctx = TimingContext(make_settings(), "cpu", 1, tmp_path / "run", observer)
    return Setup(services, FakeA7(clock, cpu_clock), FakeFill(), ctx)


def test_skipped_while_a7_is_missing(setup: Setup) -> None:
    missing = setup.target(find_spec=lambda name: None).unavailable(setup.ctx)
    assert missing is not None
    assert RUNNER_MODULE in missing and INPUTS_MODULE in missing
    only_inputs = setup.target(find_spec=lambda name: None if name == INPUTS_MODULE else object())
    assert only_inputs.unavailable(setup.ctx) == (
        f"A7 verification runner not installed ({INPUTS_MODULE})"
    )
    assert setup.target().unavailable(setup.ctx) is None


def test_wall_spans_load_and_runner_and_checks_are_timed_apart(
    setup: Setup, clip_factory: Callable[..., ClipManifest]
) -> None:
    target = setup.target()
    measures = target.run(clip_factory("c0"), setup.ctx)
    check_measures(measures)
    assert measures == {
        VERIFY_WALL_S: 0.75,
        VERIFY_LOAD_S: 0.25,
        VERIFY_CPU_S: 1.0,
        VERIFY_CHECKS_S: 0.1875,
        "verify.r4_jump.wall_s": 0.125,
        "verify.r1_pen.wall_s": 0.0625,
    }
    assert setup.a7.source.loads == 1
    assert setup.a7.runner.seen == [["c0"]]


def test_check_timing_skips_missing_inputs_and_survives_a_failing_check(setup: Setup) -> None:
    inputs = types.SimpleNamespace(contacts=None)
    checks = (
        fake_check("R2", 0.5, setup.a7.clock, requires=frozenset({"contacts"}), inputs=inputs),
        fake_check("R5a", 0.25, setup.a7.clock, fails=True, inputs=inputs),
    )
    assert setup.target()._time_checks(checks, inputs) == {"verify.r5a.wall_s": 0.25}


def test_inputs_come_from_the_persistent_cache_without_an_observer(
    setup: Setup, clip_factory: Callable[..., ClipManifest]
) -> None:
    setup.target().run(clip_factory("c0"), setup.ctx)
    [(dataset, plan, deps)] = setup.a7.sources
    assert (dataset, plan) == ("fake_ds", PLAN)
    assert deps.cache.root == setup.ctx.settings.paths.cache_dir
    assert deps.observer is None
    assert deps.lock.acquisitions == 0


def test_prepare_fills_the_cache_per_dataset_and_builds_a7_once(
    setup: Setup, clip_factory: Callable[..., ClipManifest]
) -> None:
    setup.fill.failures = (ClipFailure("c1", Stage.DEPTH, "boom"),)
    clips = [clip_factory("c0"), clip_factory("x0", dataset="other_ds"), clip_factory("c1")]
    datasets = {**setup.services.datasets, "other_ds": setup.services.datasets["fake_ds"]}
    setup.services = replace(setup.services, datasets=datasets)
    target = setup.target()
    assert target.prepare(clips, setup.ctx) == {}
    assert [ids for _, ids, _ in setup.fill.runs] == [["c0", "c1"], ["x0"]]
    for _, _, deps in setup.fill.runs:
        assert deps.observer is setup.ctx.observer
        assert deps.cache.root == setup.ctx.settings.paths.cache_dir
    info = target.describe(setup.ctx)
    assert info == {
        "input_level": InputLevel.PERCEIVED.value,
        "checks": "R4_jump,R1-pen",
        "prepare_failures": "c1 depth: boom",
    }
    target.run(clips[0], setup.ctx)
    target.run(clips[2], setup.ctx)
    assert setup.a7.builds == 1
    target.close()
    target.run(clips[0], setup.ctx)
    assert setup.a7.builds == 2


def test_failed_inputs_raise(setup: Setup, clip_factory: Callable[..., ClipManifest]) -> None:
    setup.a7 = FakeA7(setup.a7.clock, setup.a7.cpu_clock, error="no depth")
    with pytest.raises(TimingError, match="c0: verification inputs failed: no depth"):
        setup.target().run(clip_factory("c0"), setup.ctx)
    assert setup.a7.runner.seen == []


@pytest.mark.parametrize(
    ("check_id", "measure"),
    [("R4_jump", "verify.r4_jump.wall_s"), ("R3-collision", "verify.r3_collision.wall_s")],
)
def test_check_ids_become_lower_case_measures(check_id: str, measure: str) -> None:
    assert check_measure(check_id) == measure


class Section(BaseModel):
    model_config = ConfigDict(extra="allow")
    thresholds: Any = "provisional"


def fake_a7_modules(calls: dict[str, Any]) -> tuple[types.ModuleType, types.ModuleType]:
    runner = types.ModuleType(RUNNER_MODULE)
    vars(runner).update(
        VERIFICATION_SECTION="verification",
        CHECKS_SECTION="checks",
        VerificationSettings=Section,
        ChecksSettings=Section,
        build_checks=lambda registry, overrides: [CHECK],
        build_policy=lambda registry, settings: "policy",
        load_thresholds=lambda path, provisional: calls.setdefault("thresholds", path),
        VerificationRunner=lambda *args: calls.setdefault("runner", args),
    )
    inputs = types.ModuleType(INPUTS_MODULE)
    vars(inputs).update(
        PipelineOutputLoader=lambda plan, deps: ("loader", plan, deps),
        PerceivedSource=lambda *args: calls.setdefault("source", args),
    )
    return runner, inputs


def test_load_a7_wires_the_adr_api(
    monkeypatch: pytest.MonkeyPatch, make_settings: Callable[..., Settings]
) -> None:
    calls: dict[str, Any] = {}
    runner, inputs = fake_a7_modules(calls)
    monkeypatch.setitem(sys.modules, RUNNER_MODULE, runner)
    monkeypatch.setitem(sys.modules, INPUTS_MODULE, inputs)
    settings = make_settings()
    verifier = load_a7(settings, {}, {})
    assert verifier.checks == (CHECK,)
    assert calls["thresholds"] == settings.paths.thresholds
    assert calls["runner"] == ((CHECK,), "policy", settings.paths.thresholds)
    dataset: Any = types.SimpleNamespace(name="fake_ds")
    deps: Any = object()
    verifier.source(dataset, PLAN, deps)
    assert calls["source"] == (dataset, ("loader", PLAN, deps), InputLevel.PERCEIVED)


def test_load_a7_builds_the_real_a7_api_when_it_is_installed(
    make_settings: Callable[..., Settings],
) -> None:
    pytest.importorskip(RUNNER_MODULE)
    pytest.importorskip(INPUTS_MODULE)
    checks, policies = plugins.PLUGIN_KINDS["checks"](), plugins.PLUGIN_KINDS["policies"]()
    verifier = load_a7(make_settings(), checks, policies)
    assert verifier.checks
    assert {check.check_id for check in verifier.checks} <= set(checks)
    assert callable(verifier.runner.run)


def test_importing_the_targets_loads_no_heavy_library_nor_a7() -> None:
    probe = (
        "import sys\n"
        "import physverify.timing.targets.extract_pipeline\n"
        "import physverify.timing.targets.verify_checks\n"
        "import physverify.timing.targets.kubric_render\n"
        f"banned = ('torch', 'transformers', 'diffusers', 'kubric', 'bpy', {RUNNER_MODULE!r})\n"
        "print(sorted(name for name in banned if name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == "[]"
