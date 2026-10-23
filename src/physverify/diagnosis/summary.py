"""Console summary of each diagnosis level's report."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

from physverify.core.markdown import SIG3_NAN
from physverify.diagnosis.report import (
    AttributionReport,
    EndToEndReport,
    Failure,
    FallRow,
    RealVideoReport,
    StageErrorReport,
)

MAX_SUMMARY_LINES = 10


_number = SIG3_NAN


def _gravity_text(row: FallRow) -> str:
    """The estimated g with its measured relative sigma, never as a bare number."""
    sigma = "sigma unknown" if row.g_rel_sigma is None else f"rel sigma {_number(row.g_rel_sigma)}"
    return f"g [estimated] {_number(row.g_est)} m/s2 {sigma}"


def _failure_lines(failures: Sequence[Failure]) -> list[str]:
    return [f"FAILED {f.clip_id}: {f.error}" for f in failures]


@functools.singledispatch
def summary_lines(report: Any) -> list[str]:
    raise TypeError(f"no summary for {type(report).__name__}")


@summary_lines.register
def _stages_summary(report: StageErrorReport) -> list[str]:
    written = report.profile_path if report.profile_written else "not written"
    lines = [f"error profile: {written}", f"metric rows: {len(report.rows)}"]
    lines += [
        f"{r.stage.value}/{r.metric} {r.split.value}: n_clips {r.n_clips}" for r in report.rows
    ]
    return _failure_lines(report.failures) + lines


@summary_lines.register
def _end_to_end_summary(report: EndToEndReport) -> list[str]:
    lines = [
        f"{c.name} {row.check_id}: pairs {row.n_pairs} mae {_number(row.mae)} {row.unit}"
        f" flips {_number(row.flips.rate)}"
        for c in report.comparisons
        for row in c.checks
    ]
    return _failure_lines(report.failures) + lines


@summary_lines.register
def _attribution_summary(report: AttributionReport) -> list[str]:
    lines = [
        f"{c.check_id}: blocks {c.n_blocks} v0 {_number(c.v0)} {c.unit} shapley "
        + ", ".join(f"{stage.value} {_number(value)}" for stage, value in c.shapley.items())
        for c in report.checks
    ]
    return _failure_lines(report.failures) + lines


@summary_lines.register
def _real_summary(report: RealVideoReport) -> list[str]:
    lines = [
        f"{r.clip_id}: motion IoU {_number(r.motion_iou)}"
        f" false motion {_number(r.false_motion_rate)}"
        + (f" ({r.unavailable})" if r.unavailable else "")
        for r in report.motion
    ]
    lines += [
        f"{r.clip_id}: {_gravity_text(r)} closer base {r.closer_time_base}"
        + (f" abstain {r.abstain_reason.value}" if r.abstain_reason else "")
        for r in report.falls
    ]
    lines += [
        f"{v.dataset} {v.check_id}: fail rate {_number(v.fail_rate)}" for v in report.verdicts
    ]
    return _failure_lines(report.failures) + lines
