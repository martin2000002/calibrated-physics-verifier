"""Eval scores of the generator comparison under one stamp: the manifest that binds them to the
freeze, the stale-stamp check and the CPU rescore of stored check results with a new rule."""

from __future__ import annotations

from collections.abc import Sequence

from physverify.finetune.candidate_scoring import restamp, same_stamp
from physverify.finetune.layout import CampaignStore
from physverify.finetune.records import CandidateScore, write_record, write_records
from physverify.finetune.scorers._base import AggregateScorer
from physverify.generator_comparison.candidate_queries import (
    candidate_snapshot,
    eval_records,
    eval_scores,
)
from physverify.generator_comparison.prereg import ComparisonPreregistration, stamp_problem
from physverify.generator_comparison.records import ScoringManifest
from physverify.schema.checks import CheckResult
from physverify.schema.io import read_jsonl


def stale(scores: Sequence[CandidateScore], scorer: AggregateScorer) -> list[CandidateScore]:
    return [s for s in scores if not same_stamp(s, scorer)]


def manifest(
    store: CampaignStore,
    tag: str,
    scorer: AggregateScorer,
    checks_hash: str | None,
    prereg_sha256: str,
) -> ScoringManifest:
    records = eval_records(store, tag)
    return ScoringManifest(
        model_tag=tag,
        scorer=scorer.name,
        stamp=scorer.thresholds(),
        checks_hash=checks_hash,
        candidates_hash=candidate_snapshot(records).sha256,
        n=len(eval_scores(store, tag, scorer.name)),
        prereg_sha256=prereg_sha256,
    )


def rescore(store: CampaignStore, tag: str, scorer: AggregateScorer) -> int:
    """Re-aggregates every stale eval score from its stored check results; returns how many."""
    todo = stale(eval_scores(store, tag, scorer.name), scorer)
    for score in todo:
        aggregate = scorer.score(read_jsonl(CheckResult, score.results_path))
        record = restamp(score, scorer, aggregate)
        write_record(store.score_path(tag, scorer.name, score.candidate_id), record)
    write_records(store.scores_path(tag, scorer.name), store.scores(tag, scorer.name))
    return len(todo)


def frozen_stamp_problem(scorer: AggregateScorer, prereg: ComparisonPreregistration) -> str | None:
    problem = stamp_problem(scorer.version, scorer.thresholds(), prereg)
    return None if problem is None else f"the active {problem}; refreeze"
