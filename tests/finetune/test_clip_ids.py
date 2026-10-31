"""Clip ids of candidates: stable across configurations, never shared between model tags, and
claimed in the stage cache by one candidate configuration at a time."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from campaign_stores import campaign_store

from physverify.core.errors import UsageError
from physverify.config.models import Settings
from physverify.core.local_manifest import MANIFEST_FILE, QueryEntry
from physverify.core.video import write_mp4
from physverify.datasets.local_video import LocalVideoDataset
from physverify.extraction.cache import StageCache
from physverify.finetune.cache_claims import PURGE_FLAG, CacheGuard
from physverify.finetune.candidate_scoring import ScoringDeps, score_candidates
from physverify.finetune.layout import CampaignStore
from physverify.finetune.records import (
    BASE_TAG,
    CandidateRecord,
    CandidateScore,
    PromptRecord,
    write_record,
)
from physverify.finetune.scorers.provisional import ProvisionalScorer
from physverify.finetune.seeds import candidate_id
from physverify.schema.checks import CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest
from physverify.schema.stages import Stage
from physverify.verification.inputs import ClipInputs
from physverify.verification.runner import VerificationReport

PROMPT = "fall-0001"
LORA = "lora-0123456789ab"


@pytest.fixture
def local_settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    return make_settings(f"paths.raw_dir={tmp_path / 'raw'}", profile="precise")


def _prompt() -> PromptRecord:
    return PromptRecord(
        prompt_id=PROMPT,
        text="a ball falls",
        queries=[QueryEntry(phrase="ball", expected_count=1)],
        family="fall",
        template_id="fall_drop",
        target_checks=["R5a"],
        split="train",
    )


def _candidate(store: CampaignStore, tag: str, config_hash: str, k: int = 0) -> CandidateRecord:
    """A stored candidate as ``generate`` leaves it: the mp4 named by the candidate id, then the
    meta; no manifest is written."""
    video = store.video_path(tag, PROMPT, k)
    video.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(np.full((4, 16, 16, 3), 40 * (k + 1), dtype=np.uint8), video, 8.0)
    record = CandidateRecord(
        candidate_id=candidate_id(PROMPT, k),
        prompt_id=PROMPT,
        split="train",
        k=k,
        seed=k,
        model_tag=tag,
        video_path=str(video),
        latents_path=None,
        embeds_path=str(store.embeds_path(PROMPT)),
        config_hash=config_hash,
    )
    write_record(store.meta_path(tag, PROMPT, k), record)
    return record


def _cache_dir(cache: StageCache, clip_id: str) -> Path:
    return cache.directory("local_video", clip_id, Stage.DETECT, "grounding_dino", "0" * 64)


def test_base_and_lora_candidates_of_one_prompt_and_k_get_distinct_clip_ids(
    local_settings: Settings, tmp_path: Path
) -> None:
    store = campaign_store(local_settings, "c")
    base = _candidate(store, BASE_TAG, "a" * 64)
    lora = _candidate(store, LORA, "b" * 64)
    assert base.candidate_id == lora.candidate_id
    assert base.clip_id != lora.clip_id
    dataset = LocalVideoDataset(local_settings)
    read = {}
    for tag in (BASE_TAG, LORA):
        store.write_manifest(tag)
        read[tag] = [clip.clip_id for clip in dataset.clips(store.subset(tag))]
    assert read == {BASE_TAG: [base.clip_id], LORA: [lora.clip_id]}
    cache = StageCache(tmp_path / "cache")
    assert _cache_dir(cache, base.clip_id) != _cache_dir(cache, lora.clip_id)


def _guard(settings: Settings, generation: str = "g", purge: bool = False) -> CacheGuard:
    return CacheGuard(StageCache(settings.paths.cache_dir), generation, purge)


def _score(
    store: CampaignStore, runner: _Runner, guard: CacheGuard, settings: Settings
) -> list[CandidateScore]:
    deps = ScoringDeps(LocalVideoDataset(settings).clips, _Source(), runner, guard)
    return score_candidates(store, BASE_TAG, [_prompt()], ProvisionalScorer(settings, {}), deps)


def _cached(settings: Settings, clip_id: str) -> Path:
    entry = _cache_dir(StageCache(settings.paths.cache_dir), clip_id)
    entry.mkdir(parents=True, exist_ok=True)
    return entry


def test_a_base_clip_id_extracted_for_another_campaigns_configuration_is_refused(
    local_settings: Settings, make_result: Callable[..., CheckResult]
) -> None:
    runner = _Runner(make_result)
    first = _candidate(campaign_store(local_settings, "c1"), BASE_TAG, "a" * 64)
    _score(campaign_store(local_settings, "c1"), runner, _guard(local_settings), local_settings)
    entry = _cached(local_settings, first.clip_id)
    second_store = campaign_store(local_settings, "c2")
    second = _candidate(second_store, BASE_TAG, "c" * 64)
    assert first.clip_id == second.clip_id == first.candidate_id
    with pytest.raises(UsageError, match=f"another candidate configuration.*{PURGE_FLAG}"):
        _score(second_store, runner, _guard(local_settings), local_settings)
    assert entry.is_dir() and runner.clips == [first.clip_id]
    _score(second_store, runner, _guard(local_settings, purge=True), local_settings)
    assert not entry.exists() and runner.clips == [first.clip_id, second.clip_id]


def test_a_campaign_of_other_generation_settings_on_a_shared_cache_is_refused(
    local_settings: Settings, make_result: Callable[..., CheckResult]
) -> None:
    runner = _Runner(make_result)
    first_store = campaign_store(local_settings, "c1")
    first = _candidate(first_store, BASE_TAG, "a" * 64)
    _score(first_store, runner, _guard(local_settings, "g1"), local_settings)
    entry = _cached(local_settings, first.clip_id)
    second_store = campaign_store(local_settings, "c2")
    _candidate(second_store, BASE_TAG, "a" * 64, k=1)
    with pytest.raises(UsageError, match=r"campaigns \['c1'\] claim other generation settings"):
        _score(second_store, runner, _guard(local_settings, "g2"), local_settings)
    _score(second_store, runner, _guard(local_settings, "g2", purge=True), local_settings)
    assert not entry.exists()
    _candidate(first_store, BASE_TAG, "a" * 64, k=2)
    with pytest.raises(UsageError, match=r"campaigns \['c2'\]"):
        _score(first_store, runner, _guard(local_settings, "g1"), local_settings)


def test_the_same_configuration_shares_the_cache_across_campaigns(
    local_settings: Settings, make_result: Callable[..., CheckResult]
) -> None:
    runner = _Runner(make_result)
    entry = _cached(local_settings, candidate_id(PROMPT, 0))
    for campaign in ("c1", "c2"):
        store = campaign_store(local_settings, campaign)
        _candidate(store, BASE_TAG, "a" * 64)
        _score(store, runner, _guard(local_settings), local_settings)
    assert entry.is_dir() and runner.clips == [candidate_id(PROMPT, 0)] * 2


def test_an_existing_candidate_stays_complete_and_its_mp4_keeps_its_name(
    local_settings: Settings,
) -> None:
    store = campaign_store(local_settings, "c")
    record = _candidate(store, BASE_TAG, "a" * 64)
    old = store.video_dir(BASE_TAG) / MANIFEST_FILE
    old.write_text(json.dumps({"clips": [{"video": Path(record.video_path).name}]}))
    assert store.is_complete(record)
    store.write_manifest(BASE_TAG)
    assert store.is_complete(record)
    assert Path(record.video_path).name == f"{record.candidate_id}.mp4"
    assert sorted(p.name for p in store.video_dir(BASE_TAG).glob("*.mp4")) == [
        f"{record.candidate_id}.mp4"
    ]


class _Source:
    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        for clip in clips:
            yield ClipInputs(clip, None, "no perception in tests", InputLevel.PERCEIVED)


class _Runner:
    def __init__(self, make_result: Callable[..., CheckResult]) -> None:
        self.make_result = make_result
        self.clips: list[str] = []

    def run(self, source: Any, clips: Sequence[ClipManifest]) -> VerificationReport:
        ids = [item.clip.clip_id for item in source.load(clips)]
        self.clips += ids
        results = tuple(self.make_result("R1", Verdict.PASS, 0.2, 1.0, clip_id=c) for c in ids)
        return VerificationReport(results, (), tuple(clips))


def test_scoring_reads_base_candidates_of_an_old_manifest_under_their_candidate_ids(
    local_settings: Settings, make_result: Callable[..., CheckResult]
) -> None:
    store = campaign_store(local_settings, "c")
    records = [_candidate(store, BASE_TAG, "a" * 64, k) for k in range(2)]
    entries = [{"video": Path(r.video_path).name} for r in records]
    (store.video_dir(BASE_TAG) / MANIFEST_FILE).write_text(json.dumps({"clips": entries}))
    runner = _Runner(make_result)
    scores = _score(store, runner, _guard(local_settings), local_settings)
    assert runner.clips == [r.candidate_id for r in records]
    assert [s.candidate_id for s in scores] == [r.candidate_id for r in records]
    assert all(s.n_decided == 1 for s in scores)


def test_a_score_carries_its_configuration_and_another_one_rescores_it_after_a_purge(
    local_settings: Settings, make_result: Callable[..., CheckResult]
) -> None:
    store = campaign_store(local_settings, "c")
    runner = _Runner(make_result)
    _candidate(store, BASE_TAG, "a" * 64)
    (first,) = _score(store, runner, _guard(local_settings), local_settings)
    _candidate(store, BASE_TAG, "c" * 64)
    with pytest.raises(UsageError, match="another candidate configuration"):
        _score(store, runner, _guard(local_settings), local_settings)
    (second,) = _score(store, runner, _guard(local_settings, purge=True), local_settings)
    assert (first.candidate_config_hash, second.candidate_config_hash) == ("a" * 64, "c" * 64)
    assert runner.clips == [first.candidate_id, first.candidate_id]
