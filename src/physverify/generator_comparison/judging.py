"""Running a judge over stored candidates or gate videos and recording what it said (ADR 0013 D4).

The judge is injected; nothing here knows how it scores. A stored output is reused only when it
has the same stamp, the same pre-registration and the same videos.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from physverify.core.errors import UsageError
from physverify.datasets._videophy_tables import VideophyRow, to_float
from physverify.finetune.records import CandidateRecord, PromptRecord
from physverify.generator_comparison.judge_gate import gate_task
from physverify.generator_comparison.judges._base import Judge, JudgeInput, JudgeScore
from physverify.generator_comparison.records import GateReport, JudgeOutput, JudgeRow, JudgeSplit
from physverify.generator_comparison.settings import GateSettings


def candidate_items(
    records: Sequence[CandidateRecord], prompts: Mapping[str, PromptRecord]
) -> list[JudgeInput]:
    """Candidates of the given prompts, captioned with their prompt text."""
    return [
        JudgeInput(r.candidate_id, Path(r.video_path), prompts[r.prompt_id].text)
        for r in records
        if r.prompt_id in prompts
    ]


def gate_items(rows: Sequence[VideophyRow], video: Mapping[str, Path]) -> list[JudgeInput]:
    return [JudgeInput(r.clip_id, video[r.clip_id], r.caption) for r in rows]


def reusable(
    output: JudgeOutput | None, stamp: str, prereg_sha256: str | None, ids: set[str]
) -> bool:
    if output is None:
        return False
    return (output.stamp, output.prereg_sha256, set(output.rows)) == (stamp, prereg_sha256, ids)


def judge_output(
    judge: Judge,
    items: Sequence[JudgeInput],
    where: tuple[JudgeSplit, str, str],
    prereg_sha256: str | None,
) -> JudgeOutput:
    """``where`` is (split, model_tag, task)."""
    split, tag, task = where
    if not items:
        raise UsageError(f"no {split} candidates of {tag} to judge")
    scores = judge.judge(items, task)
    first = next(iter(scores.values()))
    return JudgeOutput(
        split=split,
        model_tag=tag,
        task=task,
        stamp=judge.stamp(task),
        scale=first.scale,
        prereg_sha256=prereg_sha256,
        rows={vid: JudgeRow(value=s.value, reason=s.reason) for vid, s in sorted(scores.items())},
    )


def gate_reference(
    rows: Sequence[VideophyRow], table: Mapping[str, Mapping[str, str]], column: str
) -> dict[str, float]:
    """Published AutoEval scores of the gate rows (by clip id), finite ones only."""
    found = {r.clip_id: to_float(table.get(r.video_url, {}).get(column)) for r in rows}
    return {cid: value for cid, value in found.items() if math.isfinite(value)}


def gate_report(
    judged: Mapping[str, Mapping[str, JudgeScore]],
    references: Mapping[str, Mapping[str, float]],
    stamps: Mapping[str, str],
    shas: tuple[str | None, str | None],
    cfg: GateSettings,
) -> GateReport:
    """``judged`` and ``references`` are by task; ``shas`` are (reference table, pip freeze)."""
    tasks = {
        task: gate_task(task, stamps[task], judged[task], references[task], cfg)
        for task in sorted(judged)
    }
    ids = sorted({cid for reference in references.values() for cid in reference})
    return GateReport(
        tasks=tasks,
        clip_ids=ids,
        reference_sha256=shas[0],
        freeze_sha256=shas[1],
        passed=bool(tasks) and all(t.passed for t in tasks.values()),
    )


def gate_problem(gate: GateReport | None, stamps: Mapping[str, str]) -> str | None:
    """Why the judge may not be used: no gate, a failed gate or a gate of another stamp."""
    if gate is None:
        return "the AutoEval gate has not run: run `physverify h3 gate-autoeval`"
    if not gate.passed:
        return "the AutoEval gate failed; the external judge is not used"
    for task, stamp in stamps.items():
        ran = gate.tasks.get(task)
        if ran is None or ran.stamp != stamp:
            return f"the gate did not run the {task} judge with stamp {stamp}; run it again"
    return None
