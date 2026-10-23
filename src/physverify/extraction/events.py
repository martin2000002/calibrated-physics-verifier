"""Stage events of an extraction run: the console line per stage, the recorder that keeps them
for an aborted run, the ``.jsonl`` log and the per-stage summary."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from physverify.extraction._base import StageEvent, StageObserver, StageProvider
from physverify.extraction.pipeline import ClipFailure, ExtractionReport
from physverify.extraction.plan import ResolvedPlan
from physverify.schema.stages import STAGE_ORDER


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
