"""Running a judge over stored candidates or gate videos and recording what it said (ADR 0013 D4).

The judge is injected; nothing here knows how it scores. A stored output is reused only when it
has the same stamp, the same pre-registration and the same videos.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from physverify.core.choices import choice
from physverify.core.errors import MissingInputError, UsageError
from physverify.core.output import Echo
from physverify.datasets._videophy_tables import VideophyRow, to_float
from physverify.finetune.prompts import campaign_prompts
from physverify.finetune.records import BASE_TAG, CandidateRecord, PromptRecord, write_record
from physverify.generator_comparison.arms import analysis_prompts
from physverify.generator_comparison.candidate_queries import eval_records
from physverify.generator_comparison.context import H3Context, arm_tags, selection
from physverify.generator_comparison.judge_gate import gate_task
from physverify.generator_comparison.judges._base import Judge, JudgeInput, JudgeScore
from physverify.generator_comparison.prereg import H3Preregistration
from physverify.generator_comparison.prereg_state import DEV_SPLIT
from physverify.generator_comparison.records import (
    TASKS,
    GateReport,
    JudgeOutput,
    JudgeRow,
    JudgeSplit,
    load_judge_output,
)
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
        return "the AutoEval gate has not run: run `physverify compare-generators gate-autoeval`"
    if not gate.passed:
        return "the AutoEval gate failed; the external judge is not used"
    for task, stamp in stamps.items():
        ran = gate.tasks.get(task)
        if ran is None or ran.stamp != stamp:
            return f"the gate did not run the {task} judge with stamp {stamp}; run it again"
    return None


def judge_tasks(task: str | None) -> tuple[str, ...]:
    if task is None:
        return TASKS
    return (choice(task, TASKS, "--task"),)


def dev_items(h: H3Context) -> dict[str, list[JudgeInput]]:
    """Dev candidates of the base and of the selected checkpoint (the usability test)."""
    selected = selection(h).selected
    tags = [BASE_TAG] if selected is None else [BASE_TAG, selected.model_tag]
    dev = {p.prompt_id: p for p in campaign_prompts(h.store) if p.split == DEV_SPLIT}
    return {
        tag: candidate_items([r for r in h.store.candidates(tag) if r.split == DEV_SPLIT], dev)
        for tag in tags
    }


def eval_items(h: H3Context, prereg: H3Preregistration) -> dict[str, list[JudgeInput]]:
    analysed = analysis_prompts(campaign_prompts(h.store), h.cfg.exclude_tags)
    return {
        tag: candidate_items(eval_records(h.store, tag), {p.prompt_id: p for p in analysed})
        for tag in arm_tags(prereg)
    }


def judge_one(
    h: H3Context,
    judge: Judge,
    where: tuple[JudgeSplit, str, str],
    items: list[JudgeInput],
    prereg_sha256: str | None,
    *,
    echo: Echo,
) -> None:
    path = h.layout.judge_output(*where)
    ids = {i.video_id for i in items}
    if reusable(load_judge_output(path), judge.stamp(where[2]), prereg_sha256, ids):
        echo(f"{path}: already judged")
        return
    try:
        output = judge_output(judge, items, where, prereg_sha256)
    except MissingInputError as exc:
        raise UsageError(str(exc)) from exc
    write_record(path, output)
    failed = sum(row.value is None for row in output.rows.values())
    echo(f"{where[0]}/{where[1]}/{where[2]}: {len(output.rows)} judged, {failed} unread")
