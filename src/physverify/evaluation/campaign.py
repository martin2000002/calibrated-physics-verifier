"""Store queries of H3: the eval candidates and scores of a model tag."""

from __future__ import annotations

from collections.abc import Iterable

from physverify.evaluation.arms import EVAL_SPLIT
from physverify.evaluation.prereg import CandidateSnapshot
from physverify.finetune.records import CandidateRecord, CandidateScore
from physverify.finetune.store import CampaignStore
from physverify.schema.io import stable_hash


def eval_records(store: CampaignStore, tag: str) -> list[CandidateRecord]:
    return [r for r in store.candidates(tag) if r.split == EVAL_SPLIT]


def all_eval_records(store: CampaignStore) -> list[CandidateRecord]:
    return [r for tag in store.model_tags() for r in eval_records(store, tag)]


def eval_ids(store: CampaignStore, tag: str) -> set[str]:
    return {r.candidate_id for r in eval_records(store, tag)}


def eval_scores(store: CampaignStore, tag: str, scorer: str) -> list[CandidateScore]:
    ids = eval_ids(store, tag)
    return [s for s in store.scores(tag, scorer) if s.candidate_id in ids]


def candidate_snapshot(records: Iterable[CandidateRecord]) -> CandidateSnapshot:
    rows = sorted((r.candidate_id, r.model_tag, r.config_hash) for r in records)
    return CandidateSnapshot(n=len(rows), sha256=stable_hash(rows))


def has_eval_scores(store: CampaignStore, scorers: Iterable[str]) -> bool:
    """Any eval score file of any model tag under any of ``scorers``."""
    names = list(scorers)
    return any(
        store.score_path(tag, name, cid).is_file()
        for tag in store.model_tags()
        for cid in eval_ids(store, tag)
        for name in names
    )
