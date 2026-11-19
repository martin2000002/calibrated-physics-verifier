"""The generic timing loop: repeat each target over its clips, mark warm-up calls, hand every
sample to a sink as it arrives, and isolate failures.

It knows nothing about any target. A failure in a call is recorded and the loop goes on; a
failure in a hook (construction, ``unavailable``, clip selection, ``prepare``) fails that target
only. ``close`` always runs, also on KeyboardInterrupt, which then propagates.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Protocol

from physverify.schema.clip import ClipManifest
from physverify.timing._base import TimingContext, TimingTarget
from physverify.timing.records import (
    PREPARE_S,
    CallFailure,
    Sample,
    TargetOutcome,
    TargetStatus,
    check_measures,
    unit_of,
)
from physverify.timing.settings import TargetRunSettings

IN_PROGRESS: Final = "in progress: the run stopped before this target finished"
INTERRUPTED: Final = "interrupted"
NO_CLIPS: Final = "no clips to time"

TargetFactory = Callable[[type[TimingTarget]], TimingTarget]
ClipSource = Callable[[], Sequence[ClipManifest]]
OutcomeListener = Callable[[TargetOutcome], None]

_log = logging.getLogger(__name__)


def calls_per_clip(clip_index: int, run: TargetRunSettings) -> int:
    warms = run.warmup if run.warmup_scope == "clip" or clip_index == 0 else 0
    return warms + run.repetitions


def is_warmup(clip_index: int, call_index: int, run: TargetRunSettings) -> bool:
    if run.warmup_scope == "target" and clip_index != 0:
        return False
    return call_index < run.warmup


def calls_for(n_clips: int, run: TargetRunSettings) -> int:
    """``N (W + R)`` with a per-clip warm-up, ``N R + W`` with one per target (0 without clips)."""
    return sum(calls_per_clip(index, run) for index in range(n_clips))


class SampleSink(Protocol):
    def add(self, sample: Sample) -> None: ...

    def fail(self, failure: CallFailure) -> None: ...


class _HookError(Exception):
    pass


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _hook[T](label: str, action: Callable[[], T]) -> T:
    try:
        return action()
    except Exception as exc:
        raise _HookError(f"{label}: {_error_text(exc)}") from exc


def _ignore(_outcome: TargetOutcome) -> None:
    return None


@dataclass
class _Progress:
    """What one target has done so far; becomes a TargetOutcome at any point."""

    name: str
    requires_gpu: bool
    run: TargetRunSettings
    clip_ids: tuple[str, ...] = ()
    setup: dict[str, float] = field(default_factory=dict)
    info: dict[str, str] = field(default_factory=dict)
    failures: list[CallFailure] = field(default_factory=list)
    measured_clips: set[str] = field(default_factory=set)
    n_calls: int = 0
    status: TargetStatus | None = None
    reason: str | None = None

    def stop(self, status: TargetStatus, reason: str) -> None:
        self.status, self.reason = status, reason

    def interrupt(self) -> None:
        status = TargetStatus.PARTIAL if self.measured_clips else TargetStatus.FAILED
        self.stop(status, f"{INTERRUPTED} after {self.n_calls} calls")

    def outcome(self, provisional: str | None = None) -> TargetOutcome:
        status, reason = (TargetStatus.PARTIAL, provisional) if provisional else self._final()
        return TargetOutcome(
            name=self.name,
            requires_gpu=self.requires_gpu,
            status=status,
            reason=reason,
            run=self.run,
            clip_ids=self.clip_ids,
            info=MappingProxyType(dict(self.info)),
            setup=MappingProxyType(dict(self.setup)),
            failures=tuple(self.failures),
            n_calls=self.n_calls,
        )

    def _final(self) -> tuple[TargetStatus, str | None]:
        if self.status is not None:
            return self.status, self.reason
        if not self.failures:
            return TargetStatus.MEASURED, None
        failed = f"{len(self.failures)} of {self.n_calls} calls failed"
        if self.measured_clips:
            return TargetStatus.PARTIAL, failed
        return TargetStatus.FAILED, failed


def _select(
    target: TimingTarget, clips: ClipSource, ctx: TimingContext, run: TargetRunSettings
) -> list[ClipManifest]:
    own = target.own_clips(ctx)
    chosen = list(own if own is not None else clips())
    return chosen if run.max_clips is None else chosen[: run.max_clips]


def _describe(target: TimingTarget, ctx: TimingContext) -> dict[str, str]:
    try:
        return {str(key): str(value) for key, value in target.describe(ctx).items()}
    except Exception as exc:  # noqa: BLE001 - a target's description never aborts the run
        return {"describe_error": _error_text(exc)}


def _close(target: TimingTarget, progress: _Progress) -> None:
    try:
        target.close()
    except Exception as exc:  # noqa: BLE001 - a failing close is recorded, not raised
        _log.warning("%s: close failed: %s", progress.name, _error_text(exc))
        progress.info["close_error"] = _error_text(exc)


class TimingHarness:
    """Runs targets one after another; calls are sequential, clip by clip."""

    def __init__(
        self,
        sink: SampleSink,
        clock: Callable[[], float] = time.perf_counter,
        on_outcome: OutcomeListener = _ignore,
    ) -> None:
        self._sink = sink
        self._clock = clock
        self._on_outcome = on_outcome

    def run(
        self,
        targets: Sequence[tuple[type[TimingTarget], TargetRunSettings]],
        factory: TargetFactory,
        clips: ClipSource,
        ctx: TimingContext,
    ) -> list[TargetOutcome]:
        return [self._build_and_run(cls, run, factory, clips, ctx) for cls, run in targets]

    def run_target(
        self, target: TimingTarget, run: TargetRunSettings, clips: ClipSource, ctx: TimingContext
    ) -> TargetOutcome:
        """``on_outcome`` receives a provisional outcome after ``prepare`` and the final one."""
        progress = _Progress(target.name, target.requires_gpu, run)
        try:
            self._attempt(target, progress, clips, ctx)
        except KeyboardInterrupt:
            progress.interrupt()
            raise
        finally:
            progress.info.update(_describe(target, ctx))
            _close(target, progress)
            outcome = progress.outcome()
            self._on_outcome(outcome)
        return outcome

    def _build_and_run(
        self,
        cls: type[TimingTarget],
        run: TargetRunSettings,
        factory: TargetFactory,
        clips: ClipSource,
        ctx: TimingContext,
    ) -> TargetOutcome:
        try:
            target = factory(cls)
        except Exception as exc:  # noqa: BLE001 - a target that cannot start is reported failed
            failed = TargetOutcome(
                cls.name, cls.requires_gpu, TargetStatus.FAILED, f"init: {_error_text(exc)}", run
            )
            self._on_outcome(failed)
            return failed
        return self.run_target(target, run, clips, ctx)

    def _attempt(
        self,
        target: TimingTarget,
        progress: _Progress,
        clips: ClipSource,
        ctx: TimingContext,
    ) -> None:
        try:
            reason = _hook("unavailable", lambda: target.unavailable(ctx))
            if reason is not None:
                progress.stop(TargetStatus.SKIPPED, reason)
                return
            chosen = _hook("clip selection", lambda: _select(target, clips, ctx, progress.run))
            if not chosen:
                progress.stop(TargetStatus.FAILED, NO_CLIPS)
                return
            progress.clip_ids = tuple(clip.clip_id for clip in chosen)
            progress.setup = _hook("prepare", lambda: self._prepare(target, chosen, ctx))
        except _HookError as exc:
            progress.stop(TargetStatus.FAILED, str(exc))
            return
        self._on_outcome(progress.outcome(provisional=IN_PROGRESS))
        self._loop(target, chosen, progress, ctx)

    def _prepare(
        self, target: TimingTarget, clips: Sequence[ClipManifest], ctx: TimingContext
    ) -> dict[str, float]:
        start = self._clock()
        measures = dict(target.prepare(clips, ctx))
        elapsed = self._clock() - start
        check_measures(measures)
        return {**measures, PREPARE_S: elapsed}

    def _loop(
        self,
        target: TimingTarget,
        clips: Sequence[ClipManifest],
        progress: _Progress,
        ctx: TimingContext,
    ) -> None:
        total = calls_for(len(clips), progress.run)
        for index, clip in enumerate(clips):
            for call in range(calls_per_clip(index, progress.run)):
                warmup = is_warmup(index, call, progress.run)
                _log.info(
                    "%s %s call %d of %d%s",
                    progress.name,
                    clip.clip_id,
                    progress.n_calls + 1,
                    total,
                    " (warm-up)" if warmup else "",
                )
                self._call(target, clip, call, warmup, progress, ctx)

    def _call(
        self,
        target: TimingTarget,
        clip: ClipManifest,
        call: int,
        warmup: bool,
        progress: _Progress,
        ctx: TimingContext,
    ) -> None:
        progress.n_calls += 1
        try:
            values = dict(target.run(clip, ctx))
            check_measures(values)
        except Exception as exc:  # noqa: BLE001 - a failing call is recorded and the run goes on
            failure = CallFailure(progress.name, clip.clip_id, call, _error_text(exc))
            progress.failures.append(failure)
            self._sink.fail(failure)
            return
        for measure, value in values.items():
            self._sink.add(
                Sample(
                    progress.name,
                    clip.clip_id,
                    call,
                    warmup,
                    measure,
                    unit_of(measure),
                    float(value),
                )
            )
        if values and not warmup:
            progress.measured_clips.add(clip.clip_id)
