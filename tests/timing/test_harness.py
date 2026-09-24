from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from physverify.config.models import Settings
from physverify.schema.clip import ClipManifest
from physverify.timing._base import TimingContext, TimingTarget
from physverify.timing.harness import (
    IN_PROGRESS,
    NO_CLIPS,
    TimingHarness,
    calls_for,
    is_warmup,
)
from physverify.timing.records import (
    PREPARE_S,
    CallFailure,
    Sample,
    TargetOutcome,
    TargetStatus,
)
from physverify.timing.services import TimingServices
from physverify.timing.settings import TargetRunSettings
from physverify.timing.stats import per_clip

ONE_CALL = TargetRunSettings(repetitions=1)


class ListSink:
    def __init__(self) -> None:
        self.samples: list[Sample] = []
        self.failures: list[CallFailure] = []

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)

    def fail(self, failure: CallFailure) -> None:
        self.failures.append(failure)


@pytest.fixture
def ctx(make_settings: Callable[..., Settings], tmp_path: Path) -> TimingContext:
    return TimingContext(make_settings(), "cpu", 1, tmp_path)


@pytest.fixture
def clips(clip_factory: Callable[..., ClipManifest]) -> list[ClipManifest]:
    return [clip_factory(clip_id) for clip_id in ("c0", "c1", "c2")]


@pytest.fixture
def build(make_services: Callable[..., TimingServices]) -> Callable[[type[Any]], TimingTarget]:
    services = make_services()

    def construct(cls: type[Any]) -> TimingTarget:
        target: TimingTarget = cls(services)
        return target

    return construct


def run_one(
    target: TimingTarget,
    run: TargetRunSettings,
    clips: Sequence[ClipManifest],
    ctx: TimingContext,
    sink: ListSink | None = None,
    outcomes: list[TargetOutcome] | None = None,
    clock: Callable[[], float] | None = None,
) -> TargetOutcome:
    listener = (outcomes if outcomes is not None else []).append
    harness = TimingHarness(
        sink or ListSink(), on_outcome=listener, **({"clock": clock} if clock else {})
    )
    return harness.run_target(target, run, lambda: clips, ctx)


def raising(message: str) -> Callable[..., Any]:
    def hook(*args: Any) -> Any:
        raise RuntimeError(message)

    return hook


@pytest.mark.parametrize(("scope", "expected"), [("clip", 3 * (2 + 3)), ("target", 3 * 3 + 2)])
def test_the_number_of_calls_follows_calls_for(
    scope: str,
    expected: int,
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    run = TargetRunSettings(warmup=2, warmup_scope=scope, repetitions=3)
    cls = make_fake_target()
    sink = ListSink()
    outcome = run_one(build(cls), run, clips, ctx, sink)
    assert calls_for(len(clips), run) == expected == len(cls.journal.calls) == outcome.n_calls
    warm = [s for s in sink.samples if s.warmup]
    assert len(warm) == (2 * 3 if scope == "clip" else 2)
    assert outcome.status is TargetStatus.MEASURED


def test_calls_for_without_clips_is_zero() -> None:
    assert calls_for(0, TargetRunSettings(warmup=1, warmup_scope="target")) == 0


def test_is_warmup_per_scope() -> None:
    per_target = TargetRunSettings(warmup=1, warmup_scope="target")
    per_clip_run = TargetRunSettings(warmup=1, warmup_scope="clip")
    assert is_warmup(0, 0, per_target) and not is_warmup(1, 0, per_target)
    assert is_warmup(1, 0, per_clip_run) and not is_warmup(1, 1, per_clip_run)


def test_warmup_value_is_excluded_from_the_median(
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    script = [{"fake.wall_s": v} for v in (100.0, 1.0, 2.0, 3.0)]
    sink = ListSink()
    run = TargetRunSettings(warmup=1, warmup_scope="clip", repetitions=3)
    run_one(build(make_fake_target(script=script)), run, clips[:1], ctx, sink)
    assert [s.warmup for s in sink.samples] == [True, False, False, False]
    assert [s.repetition for s in sink.samples] == [0, 1, 2, 3]
    (row,) = per_clip(sink.samples, 0.1)
    assert row.median == 2.0


def test_a_failing_call_is_recorded_and_the_loop_goes_on(
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    script = [{"fake.wall_s": 1.0}, RuntimeError("boom"), {"fake.wall_s": 2.0}]
    cls = make_fake_target(script=script)
    sink = ListSink()
    outcome = run_one(build(cls), ONE_CALL, clips, ctx, sink)
    assert cls.journal.calls == ["c0", "c1", "c2"]
    assert sink.failures == [CallFailure("fake", "c1", 0, "RuntimeError: boom")]
    assert outcome.failures == tuple(sink.failures)
    assert outcome.status is TargetStatus.PARTIAL
    assert cls.journal.closed == 1


def test_every_call_failing_fails_the_target(
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    cls = make_fake_target(script=[ValueError("no")])
    outcome = run_one(build(cls), ONE_CALL, clips, ctx)
    assert outcome.status is TargetStatus.FAILED
    assert outcome.reason == "3 of 3 calls failed"
    assert cls.journal.closed == 1


@pytest.mark.parametrize(
    "values", [{"fake.wall_s": math.nan}, {"fake.wall": 1.0}, {"fake.wall_s": -1.0}]
)
def test_an_invalid_measure_fails_that_call_only(
    values: Mapping[str, float],
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    sink = ListSink()
    cls = make_fake_target(script=[values, {"fake.wall_s": 1.0}])
    outcome = run_one(build(cls), ONE_CALL, clips, ctx, sink)
    assert [f.clip_id for f in sink.failures] == ["c0"]
    assert "TimingError" in sink.failures[0].error
    assert [s.clip_id for s in sink.samples] == ["c1", "c2"]
    assert outcome.status is TargetStatus.PARTIAL


def test_unavailable_target_is_skipped_without_calls(
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    cls = make_fake_target(unavailable="no weights")
    outcome = run_one(build(cls), ONE_CALL, clips, ctx)
    assert (outcome.status, outcome.reason) == (TargetStatus.SKIPPED, "no weights")
    assert cls.journal.calls == [] and cls.journal.closed == 1


def test_prepare_measures_go_to_setup_with_the_prepare_time(
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    ticks = iter([10.0, 12.5])
    cls = make_fake_target(setup={"generate.load_s": 4.0}, info={"model": "fake"})
    outcome = run_one(build(cls), ONE_CALL, clips, ctx, clock=lambda: next(ticks))
    assert dict(outcome.setup) == {"generate.load_s": 4.0, PREPARE_S: 2.5}
    assert dict(outcome.info) == {"model": "fake"}
    assert cls.journal.prepared == [("c0", "c1", "c2")]
    assert outcome.clip_ids == ("c0", "c1", "c2")


@pytest.mark.parametrize(
    ("hook", "label"),
    [
        ("unavailable", "unavailable"),
        ("own_clips", "clip selection"),
        ("prepare", "prepare"),
    ],
)
def test_a_failing_hook_fails_only_that_target(
    hook: str,
    label: str,
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    cls = make_fake_target()
    broken = type("Broken", (cls,), {hook: raising("hook broke")})
    outcome = run_one(build(broken), ONE_CALL, clips, ctx)
    assert outcome.status is TargetStatus.FAILED
    assert outcome.reason == f"{label}: RuntimeError: hook broke"
    assert cls.journal.calls == [] and cls.journal.closed == 1


def test_a_failing_clip_source_fails_only_that_target(
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    ctx: TimingContext,
) -> None:
    cls = make_fake_target()
    harness = TimingHarness(ListSink())
    outcome = harness.run_target(build(cls), ONE_CALL, raising("no dataset"), ctx)
    assert outcome.reason == "clip selection: RuntimeError: no dataset"
    assert cls.journal.closed == 1


def test_an_empty_selection_fails_the_target(
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    ctx: TimingContext,
) -> None:
    outcome = run_one(build(make_fake_target()), ONE_CALL, [], ctx)
    assert (outcome.status, outcome.reason) == (TargetStatus.FAILED, NO_CLIPS)


def test_invalid_setup_measures_fail_prepare(
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    outcome = run_one(build(make_fake_target(setup={"load": 1.0})), ONE_CALL, clips, ctx)
    assert outcome.status is TargetStatus.FAILED
    assert outcome.reason is not None and outcome.reason.startswith("prepare: TimingError")


def test_own_clips_replace_the_selection_and_max_clips_applies(
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clip_factory: Callable[..., ClipManifest],
    ctx: TimingContext,
) -> None:
    own = [clip_factory(f"p{i}", dataset="prompts") for i in range(4)]
    cls = make_fake_target(own_clips=own)
    harness = TimingHarness(ListSink())
    run = TargetRunSettings(repetitions=1, max_clips=2)
    outcome = harness.run_target(build(cls), run, raising("selection must not be used"), ctx)
    assert outcome.clip_ids == ("p0", "p1")
    assert cls.journal.calls == ["p0", "p1"]


def test_describe_and_close_errors_are_kept_in_info(
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    cls = make_fake_target()
    broken = type("Broken", (cls,), {"describe": raising("d"), "close": raising("c")})
    outcome = run_one(build(broken), ONE_CALL, clips, ctx)
    assert outcome.status is TargetStatus.MEASURED
    assert dict(outcome.info) == {
        "describe_error": "RuntimeError: d",
        "close_error": "RuntimeError: c",
    }


def test_listener_gets_a_provisional_then_the_final_outcome(
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    outcomes: list[TargetOutcome] = []
    run_one(build(make_fake_target()), ONE_CALL, clips, ctx, outcomes=outcomes)
    assert [(o.status, o.reason) for o in outcomes] == [
        (TargetStatus.PARTIAL, IN_PROGRESS),
        (TargetStatus.MEASURED, None),
    ]


def test_keyboard_interrupt_closes_the_target_and_propagates(
    make_fake_target: Callable[..., type[Any]],
    build: Callable[[type[Any]], TimingTarget],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    cls = make_fake_target(script=[{"fake.wall_s": 1.0}, KeyboardInterrupt()])
    sink = ListSink()
    outcomes: list[TargetOutcome] = []
    with pytest.raises(KeyboardInterrupt):
        run_one(build(cls), ONE_CALL, clips, ctx, sink, outcomes)
    assert cls.journal.closed == 1
    assert len(sink.samples) == 1
    final = outcomes[-1]
    assert final.status is TargetStatus.PARTIAL
    assert final.reason == "interrupted after 2 calls"


def test_run_builds_each_target_and_isolates_constructor_errors(
    make_fake_target: Callable[..., type[Any]],
    make_services: Callable[..., TimingServices],
    clips: list[ClipManifest],
    ctx: TimingContext,
) -> None:
    good, bad = make_fake_target("good"), make_fake_target("bad")

    def factory(cls: type[Any]) -> TimingTarget:
        if cls is bad:
            raise OSError("cannot build")
        target: TimingTarget = cls(make_services())
        return target

    outcomes: list[TargetOutcome] = []
    harness = TimingHarness(ListSink(), on_outcome=outcomes.append)
    results = harness.run([(bad, ONE_CALL), (good, ONE_CALL)], factory, lambda: clips, ctx)
    assert [(r.name, r.status) for r in results] == [
        ("bad", TargetStatus.FAILED),
        ("good", TargetStatus.MEASURED),
    ]
    assert results[0].reason == "init: OSError: cannot build"
    assert outcomes[0] == results[0] and outcomes[-1] == results[1]
