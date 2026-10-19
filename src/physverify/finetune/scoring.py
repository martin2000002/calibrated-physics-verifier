"""Score the candidates of one model tag: extraction + checks + clip score (ADR 0012, 3.6).

The perceived inputs come from the normal extraction pipeline over the campaign's ``local_video``
subset; the checks and their verdicts from ``VerificationRunner``. Each candidate writes its
check results and then its ``CandidateScore`` (the completion mark), so a rerun skips it unless
the score is stale: another scorer stamp, or another candidate ``config_hash`` (or none recorded).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.core.registry import Registry, discover
from physverify.finetune.records import (
    CandidateRecord,
    CandidateScore,
    CheckSummary,
    PromptRecord,
    read_record,
    write_record,
    write_records,
)
from physverify.finetune.scorers._base import AggregateScore, AggregateScorer
from physverify.finetune.seeds import WHOLE, Shard
from physverify.finetune.signals import motion_index, objects_ok
from physverify.finetune.store import CampaignStore
from physverify.schema.checks import CheckResult
from physverify.schema.clip import ClipManifest
from physverify.schema.io import read_jsonl, write_jsonl
from physverify.schema.states import ObjectStateTable
from physverify.verification.inputs import ClipInputs, InputSource
from physverify.verification.runner import TableRow, VerificationReport

SCORERS_PACKAGE: Final = "physverify.finetune.scorers"
CHUNK: Final = 8


def scorer_registry() -> Registry[type[AggregateScorer]]:
    return discover(SCORERS_PACKAGE, (AggregateScorer,), "name")


def build_scorer(name: str, settings: Settings) -> AggregateScorer:
    registry = scorer_registry()
    if name not in registry:
        raise UsageError(f"unknown scorer {name!r}; known: {sorted(registry)}")
    return registry[name](settings)


class Runner(Protocol):
    def run(self, source: InputSource, clips: Sequence[ClipManifest]) -> VerificationReport: ...


@dataclass(frozen=True)
class ScoringDeps:
    """``clips(subset)`` lists the campaign's clips, ``source`` gives their perceived inputs."""

    clips: Callable[[str], Sequence[ClipManifest]]
    source: InputSource
    runner: Runner


@dataclass
class RecordingSource:
    """Passes the inputs through and keeps the states of each clip for the signals."""

    inner: InputSource
    states: dict[str, ObjectStateTable] = field(default_factory=dict)

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        for item in self.inner.load(clips):
            if item.inputs is not None:
                self.states[item.clip.clip_id] = item.inputs.states
            yield item


def summaries(rows: Sequence[TableRow]) -> dict[str, CheckSummary]:
    return {
        row.check_id: CheckSummary(
            status=row.status.value,
            magnitude=row.magnitude,
            sigma=row.sigma,
            verdict=None if row.verdict is None else row.verdict.value,
            unit=row.unit,
        )
        for row in rows
    }


def candidate_score(
    record: CandidateRecord,
    prompt: PromptRecord,
    scorer: AggregateScorer,
    aggregate: AggregateScore,
    rows: Sequence[TableRow],
    states: ObjectStateTable | None,
    results_path: str,
) -> CandidateScore:
    return CandidateScore(
        scorer=scorer.name,
        scorer_version=scorer.version,
        calibrated=scorer.calibrated,
        thresholds=scorer.thresholds(),
        candidate_id=record.candidate_id,
        prompt_id=record.prompt_id,
        model_tag=record.model_tag,
        candidate_config_hash=record.config_hash,
        score=aggregate.score,
        sigma=aggregate.sigma,
        abstained=aggregate.abstained or aggregate.score is None,
        reason=aggregate.reason,
        n_decided=aggregate.n_decided,
        n_failed=aggregate.n_failed,
        coverage=aggregate.coverage,
        decided=sorted(aggregate.decided),
        per_check=summaries(rows),
        motion=None if states is None else motion_index(states),
        objects_ok=None if states is None else objects_ok(states, prompt.queries),
        results_path=results_path,
    )


def restamp(
    score: CandidateScore, scorer: AggregateScorer, aggregate: AggregateScore
) -> CandidateScore:
    """The stored score under a new scorer stamp; the check summaries and signals are kept."""
    return score.model_copy(
        update={
            "scorer": scorer.name,
            "scorer_version": scorer.version,
            "calibrated": scorer.calibrated,
            "thresholds": scorer.thresholds(),
            "score": aggregate.score,
            "sigma": aggregate.sigma,
            "abstained": aggregate.abstained or aggregate.score is None,
            "reason": aggregate.reason,
            "n_decided": aggregate.n_decided,
            "n_failed": aggregate.n_failed,
            "coverage": aggregate.coverage,
            "decided": sorted(aggregate.decided),
        }
    )


def same_stamp(score: CandidateScore, scorer: AggregateScorer) -> bool:
    return (score.scorer_version, score.thresholds) == (scorer.version, scorer.thresholds())


def is_current(score: CandidateScore, scorer: AggregateScorer, config_hash: str | None) -> bool:
    """The score is of the active scorer stamp and of the candidate as it is stored now."""
    hashed = score.candidate_config_hash is not None
    return hashed and score.candidate_config_hash == config_hash and same_stamp(score, scorer)


def stale_scores(
    scores: Sequence[CandidateScore],
    scorer: AggregateScorer,
    candidates: Sequence[CandidateRecord],
) -> list[str]:
    """Candidate ids whose score is not current (a score without a candidate is stale too)."""
    hashes = {record.candidate_id: record.config_hash for record in candidates}
    return sorted(
        s.candidate_id for s in scores if not is_current(s, scorer, hashes.get(s.candidate_id))
    )


def _scored(store: CampaignStore, record: CandidateRecord, scorer: AggregateScorer) -> bool:
    path = store.score_path(record.model_tag, scorer.name, record.candidate_id)
    if not path.is_file():
        return False
    return is_current(read_record(path, CandidateScore), scorer, record.config_hash)


def pending(
    store: CampaignStore,
    model_tag: str,
    scorer: AggregateScorer,
    prompts: Mapping[str, PromptRecord],
    shard: Shard,
) -> list[CandidateRecord]:
    return [
        record
        for record in store.candidates(model_tag)
        if record.prompt_id in prompts
        and shard.owns(record.candidate_id)
        and not _scored(store, record, scorer)
    ]


def score_candidates(
    store: CampaignStore,
    model_tag: str,
    prompts: Sequence[PromptRecord],
    scorer: AggregateScorer,
    deps: ScoringDeps,
    shard: Shard = WHOLE,
) -> list[CandidateScore]:
    """Scores every candidate of ``prompts`` without a current score; returns all scores of the
    tag."""
    reason = scorer.unavailable()
    if reason is not None:
        raise UsageError(f"scorer {scorer.name} is unavailable: {reason}")
    by_id = {prompt.prompt_id: prompt for prompt in prompts}
    todo = pending(store, model_tag, scorer, by_id, shard)
    if todo:
        clips = {clip.clip_id: clip for clip in deps.clips(store.subset(model_tag))}
        missing = sorted(r.candidate_id for r in todo if r.candidate_id not in clips)
        if missing:
            raise UsageError(f"candidates missing from the local_video manifest: {missing[:5]}")
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start : start + CHUNK]
            _score_chunk(store, chunk, [clips[r.candidate_id] for r in chunk], by_id, scorer, deps)
    scores = store.scores(model_tag, scorer.name)
    write_records(store.scores_path(model_tag, scorer.name), scores)
    return scores


def _score_chunk(
    store: CampaignStore,
    records: Sequence[CandidateRecord],
    clips: Sequence[ClipManifest],
    prompts: Mapping[str, PromptRecord],
    scorer: AggregateScorer,
    deps: ScoringDeps,
) -> None:
    source = RecordingSource(deps.source)
    report = deps.runner.run(source, clips)
    for record in records:
        cid = record.candidate_id
        results = [r for r in report.results if r.clip_id == cid]
        rows = [row for row in report.table if row.clip_id == cid]
        checks = store.checks_path(record.model_tag, scorer.name, cid)
        checks.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(checks, results)
        score = candidate_score(
            record,
            prompts[record.prompt_id],
            scorer,
            scorer.score(results),
            rows,
            source.states.get(cid),
            str(checks),
        )
        write_record(store.score_path(record.model_tag, scorer.name, cid), score)


Rescore = Callable[[CandidateScore, frozenset[str]], AggregateScore]


def rescorer(scorer: AggregateScorer) -> Rescore:
    """Recomputes a stored candidate's score on a subset of its checks (``only``)."""

    def rescore(score: CandidateScore, only: frozenset[str]) -> AggregateScore:
        results = read_jsonl(CheckResult, score.results_path)
        return scorer.score(results, only=only)

    return rescore
