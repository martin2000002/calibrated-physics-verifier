"""Markdown sections of the real-video diagnosis report: motion agreement, free fall, verdicts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from physverify.diagnosis.markdown_cells import section, text_table
from physverify.diagnosis.report import PROVISIONAL_NOTE, FallRow, MotionRow, VerdictDistribution
from physverify.schema.checks import AbstainReason, Verdict


def motion_section(rows: Sequence[MotionRow]) -> list[str]:
    headers = (
        "dataset",
        "clip",
        "scenario",
        "dynamic",
        "frames",
        "motion IoU",
        "precision",
        "recall",
        "false motion",
        "unavailable",
    )
    body = [
        (
            r.dataset,
            r.clip_id,
            r.scenario,
            r.dynamic,
            r.n_frames,
            r.motion_iou,
            r.precision,
            r.recall,
            r.false_motion_rate,
            r.unavailable,
        )
        for r in rows
    ]
    note = "Physics-IQ masks mark pixel change, not objects: IoU is agreement, not accuracy."
    return section("Motion agreement", [note, "", *text_table(headers, body)] if body else [])


def fall_section(rows: Sequence[FallRow]) -> list[str]:
    headers = (
        "dataset",
        "clip",
        "object",
        "frames",
        "residual ratio",
        "lateral ratio",
        "g [estimated]",
        "g rel error [estimated]",
        "g rel sigma",
        "g declared dt",
        "g container dt",
        "closer",
        "gravity dir",
        "abstain",
    )
    body = [
        (
            r.dataset,
            r.clip_id,
            r.object_id,
            r.n_frames,
            r.residual_ratio,
            r.lateral_ratio,
            r.g_est,
            r.g_rel_error,
            r.g_rel_sigma,
            r.g_declared,
            r.g_container,
            r.closer_time_base,
            "[estimated] camera +y" if r.gravity_dir_estimated else "states",
            r.abstain_reason,
        )
        for r in rows
    ]
    return section("Free fall", text_table(headers, body) if body else [])


def verdict_section(rows: Sequence[VerdictDistribution], provisional: bool) -> list[str]:
    verdicts = tuple(Verdict)
    headers = ("dataset", "check", *(v.value for v in verdicts), "fail rate", "abstentions")
    body = [
        (
            r.dataset,
            r.check_id,
            *(r.counts.get(v, 0) for v in verdicts),
            r.fail_rate,
            _counts_text(r.abstentions),
        )
        for r in rows
    ]
    note = f"Thresholds are {PROVISIONAL_NOTE}." if provisional else "Thresholds are calibrated."
    return section("Verdicts on real video", [note, "", *text_table(headers, body)] if body else [])


def _counts_text(counts: Mapping[AbstainReason, int]) -> str:
    return ", ".join(f"{r.value} {n}" for r, n in sorted(counts.items())) or "-"
