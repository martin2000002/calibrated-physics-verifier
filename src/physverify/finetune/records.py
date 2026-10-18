"""Records of the preference cycle (ADR 0012, 3.2): frozen pydantic models, one JSON line each.

A score lies in [0, 1], higher is more consistent; an abstention is ``score = None``, never 0 or 1.
NaN is never stored: an unknown sigma is ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from physverify.datasets.local_video import QueryList
from physverify.finetune.settings import PromptSplit
from physverify.synth.store import atomic_write_bytes

BASE_TAG = "base"
DEMO_TAG = "demo"
VAL_TAG = "val"
UNSEEN_TAG = "unseen_template"
TEMPLATE_TAG_PREFIX = "template:"


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def finite_or_none(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


class PromptRecord(_Record):
    prompt_id: str
    text: str
    queries: QueryList
    family: str
    template_id: str
    target_checks: list[str]
    split: PromptSplit
    tags: list[str] = Field(default_factory=list)


class CandidateRecord(_Record):
    candidate_id: str
    prompt_id: str
    split: PromptSplit
    k: int = Field(ge=0)
    seed: int = Field(ge=0)
    model_tag: str
    video_path: str
    latents_path: str | None
    embeds_path: str
    gen_s: float | None = None
    commit: str = ""
    config_hash: str


class CheckSummary(_Record):
    """One table row of the verifier for a candidate (``verification.runner.TableRow``)."""

    status: str
    magnitude: float | None
    sigma: float | None
    verdict: str | None
    unit: str

    @field_validator("magnitude", "sigma", mode="before")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        return finite_or_none(value)


class ScorerStamp(_Record):
    """Which verifier produced a record; every score, pair and report carries these (D7)."""

    scorer: str
    scorer_version: str
    calibrated: bool
    thresholds: str


class CandidateScore(ScorerStamp):
    candidate_id: str
    prompt_id: str
    model_tag: str
    score: float | None = Field(None, ge=0, le=1)
    sigma: float | None = Field(None, ge=0)
    abstained: bool
    reason: str | None
    n_decided: int = Field(ge=0)
    n_failed: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)
    decided: list[str]
    per_check: dict[str, CheckSummary]
    motion: float | None
    objects_ok: bool | None
    results_path: str

    @field_validator("score", "sigma", "motion", mode="before")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        return finite_or_none(value)


class PairRecord(ScorerStamp):
    pair_id: str
    prompt_id: str
    winner: str
    loser: str
    score_w: float
    score_l: float
    margin: float
    weight: float = Field(gt=0, le=1)
    common_checks: list[str]


class PairsReport(ScorerStamp):
    prompts: int
    candidates: int
    eligible: int
    pairs: int
    dropped: dict[str, int]
    rejected: dict[str, int]
    yield_per_prompt: float


def write_records(path: Path, records: Iterable[BaseModel]) -> None:
    """JSON lines, written atomically."""
    text = "".join(record.model_dump_json() + "\n" for record in records)
    atomic_write_bytes(path, text.encode("utf-8"))


def read_records[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line.strip()]


def write_record(path: Path, record: BaseModel) -> None:
    atomic_write_bytes(path, record.model_dump_json(indent=2).encode("utf-8"))


def read_record[M: BaseModel](path: Path, model: type[M]) -> M:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def by_prompt(scores: Sequence[CandidateScore]) -> dict[str, list[CandidateScore]]:
    grouped: dict[str, list[CandidateScore]] = {}
    for score in scores:
        grouped.setdefault(score.prompt_id, []).append(score)
    return grouped
