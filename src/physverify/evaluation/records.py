"""On-disk records of H3: scoring manifests, judge outputs and the gate (ADR 0013)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from physverify.core.models import FrozenModel
from physverify.finetune.records import read_record

JudgeSplit = Literal["dev", "eval"]
JUDGE_SPLITS: tuple[JudgeSplit, ...] = ("dev", "eval")
JudgeKey = tuple[str, str]
"""(model_tag, video_id): a base and a fine-tuned candidate of one seed share the candidate id."""
TASKS: tuple[str, ...] = ("pc", "sa")


class ScoringManifest(FrozenModel):
    """Written by ``h3 score`` and ``h3 rescore``; binds a tag's eval scores to a freeze."""

    model_tag: str
    scorer: str
    stamp: str
    checks_hash: str | None
    candidates_hash: str
    n: int
    prereg_sha256: str


class JudgeRow(FrozenModel):
    value: float | None
    reason: str | None = None


class JudgeOutput(FrozenModel):
    split: JudgeSplit
    model_tag: str
    task: str
    stamp: str
    scale: tuple[float, float]
    prereg_sha256: str | None
    rows: dict[str, JudgeRow]


class GateTask(FrozenModel):
    task: str
    stamp: str
    n: int
    exact: int
    exact_share: float
    mae: float | None
    parse_failures: int
    passed: bool


class GateReport(FrozenModel):
    tasks: dict[str, GateTask]
    clip_ids: list[str]
    reference_sha256: str | None
    freeze_sha256: str | None
    passed: bool


class TaskUsability(FrozenModel):
    n: int
    parse_fail_share: float
    modal_share: float | None
    sd: float | None
    usable: bool


class Usability(FrozenModel):
    tasks: dict[str, TaskUsability] = Field(default_factory=dict)
    usable: bool


def load_judge_output(path: Path) -> JudgeOutput | None:
    return read_record(path, JudgeOutput) if path.is_file() else None


def judge_values(output: JudgeOutput | None) -> dict[str, float | None]:
    return {} if output is None else {vid: row.value for vid, row in output.rows.items()}


def keyed_values(output: JudgeOutput | None) -> dict[JudgeKey, float | None]:
    if output is None:
        return {}
    return {(output.model_tag, vid): row.value for vid, row in output.rows.items()}
