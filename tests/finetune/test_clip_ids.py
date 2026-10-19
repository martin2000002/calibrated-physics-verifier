"""Clip ids of candidates: one per (candidate, configuration), never shared through the cache."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from physverify.config.models import Settings
from physverify.core.video import write_mp4
from physverify.datasets.local_video import MANIFEST_FILE, LocalVideoDataset, QueryEntry
from physverify.extraction.cache import StageCache
from physverify.finetune.records import BASE_TAG, CandidateRecord, PromptRecord, write_record
from physverify.finetune.scorers.provisional import ProvisionalScorer
from physverify.finetune.scoring import ScoringDeps, score_candidates
from physverify.finetune.seeds import candidate_id
from physverify.finetune.store import CampaignStore
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


def _store(settings: Settings, campaign: str = "c") -> CampaignStore:
    return CampaignStore(
        settings.paths.data_dir / "finetune" / campaign,
        settings.paths.raw_dir / "local_video",
        campaign,
    )


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
    store = _store(local_settings)
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


def test_the_same_candidate_in_another_campaign_or_configuration_gets_another_clip_id(
    local_settings: Settings,
) -> None:
    first = _candidate(_store(local_settings, "c1"), BASE_TAG, "a" * 64)
    second = _candidate(_store(local_settings, "c2"), BASE_TAG, "c" * 64)
    assert first.candidate_id == second.candidate_id
    assert first.clip_id != second.clip_id


def test_an_existing_candidate_stays_complete_and_its_mp4_keeps_its_name(
    local_settings: Settings,
) -> None:
    store = _store(local_settings)
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


def test_scoring_reads_an_old_manifest_under_the_new_clip_ids(
    local_settings: Settings, make_result: Callable[..., CheckResult]
) -> None:
    store = _store(local_settings)
    records = [_candidate(store, BASE_TAG, "a" * 64, k) for k in range(2)]
    entries = [{"video": Path(r.video_path).name} for r in records]
    (store.video_dir(BASE_TAG) / MANIFEST_FILE).write_text(json.dumps({"clips": entries}))
    runner = _Runner(make_result)
    deps = ScoringDeps(LocalVideoDataset(local_settings).clips, _Source(), runner)
    scores = score_candidates(store, BASE_TAG, [_prompt()], ProvisionalScorer(local_settings), deps)
    assert runner.clips == [r.clip_id for r in records]
    assert [s.candidate_id for s in scores] == [r.candidate_id for r in records]
    assert all(s.n_decided == 1 for s in scores)
