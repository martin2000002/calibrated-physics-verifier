"""Rule 1.1-v2 changed what a calibrated score means: weight-0 checks no longer vote, so a clip
decided only by them abstains, and scores sealed by the previous scorer version are rescored."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

CALIBRATION_TESTS = str(Path(__file__).resolve().parent.parent / "calibration")
if CALIBRATION_TESTS not in sys.path:
    sys.path.insert(0, CALIBRATION_TESTS)

from stratum_cases import (  # noqa: E402
    artifact_of,
    check_calibration,
    stratum_model,
    write_artifact,
)
from wiring import active_checks_hash, checks  # noqa: E402

from physverify.app.finetune import cache_guard  # noqa: E402
from physverify.calibration.apply import NO_WEIGHTED  # noqa: E402
from physverify.calibration.artifact_io import default_artifact_path  # noqa: E402
from physverify.config.models import Settings  # noqa: E402
from physverify.core.local_manifest import QueryEntry  # noqa: E402
from physverify.finetune.candidate_scoring import (  # noqa: E402
    ScoringDeps,
    score_candidates,
    stale_scores,
)
from physverify.finetune.layout import CampaignStore  # noqa: E402
from physverify.finetune.pairs import ABSTAINED, form_pairs, stamp_of  # noqa: E402
from physverify.finetune.records import (  # noqa: E402
    BASE_TAG,
    CandidateRecord,
    CandidateScore,
    PromptRecord,
    read_record,
    write_record,
)
from physverify.finetune.scorers.calibrated import CalibratedScorer  # noqa: E402
from physverify.finetune.settings import load_finetune  # noqa: E402
from physverify.schema.checks import CheckResult, InputLevel, Verdict  # noqa: E402
from physverify.schema.clip import ClipManifest, Split, TimeSource  # noqa: E402
from physverify.verification.inputs import ClipInputs  # noqa: E402
from physverify.verification.runner import VerificationReport  # noqa: E402

Result = Callable[..., CheckResult]
Score = Callable[..., CandidateScore]
PROMPT = "fall-0001"
PREVIOUS_VERSION = "2"
"""The version that sealed scores under rule 1.1 (weight-0 checks voted)."""


def _weightless_r1_artifact(settings: Settings) -> None:
    calibrated = {"R1": check_calibration(tau=1.0, weight=0.0), "R2": check_calibration(tau=1.0)}
    stratum = stratum_model(calibrated, InputLevel.PERCEIVED, 0.5, settings.profile)
    artifact = artifact_of((stratum,), checks_hash=active_checks_hash(settings))
    path = default_artifact_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.to_json(), encoding="utf-8")


def test_a_clip_decided_only_by_weightless_checks_abstains_and_gives_no_pair(
    settings: Settings, make_result: Result, make_score: Score
) -> None:
    _weightless_r1_artifact(settings)
    scorer = CalibratedScorer(settings, checks())
    results = [
        make_result("R1", Verdict.PASS, 0.1, 1.0, 0.01),
        make_result("R2", Verdict.ABSTAIN, None, None),
    ]
    score = scorer.score(results)
    assert (score.abstained, score.reason, score.score) == (True, NO_WEIGHTED, None)
    assert score.decided == frozenset()
    stamp = {"scorer": scorer.name, "scorer_version": scorer.version, "calibrated": True}
    candidates = [
        make_score(
            f"{PROMPT}__k{k:02d}",
            None,
            thresholds=scorer.thresholds(),
            reason=score.reason,
            n_decided=0,
            decided=[],
            **stamp,
        )
        for k in range(2)
    ]
    rules = load_finetune(settings).pairs
    formed = form_pairs(candidates, rules, lambda s, only: score, stamp_of(candidates))
    assert formed.pairs == [] and formed.dropped == {ABSTAINED: 2}


def _clip(clip_id: str) -> ClipManifest:
    return ClipManifest(
        dataset="local_video",
        clip_id=clip_id,
        split=Split.EVAL,
        video_path=f"{clip_id}.mp4",
        n_frames=4,
        width=16,
        height=16,
        container_fps=16.0,
        seconds_per_frame=1 / 16,
        time_source=TimeSource.CONTAINER_FPS,
    )


class _Source:
    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        for clip in clips:
            yield ClipInputs(clip, None, "no perception in tests", InputLevel.PERCEIVED)


class _Runner:
    def __init__(self, make_result: Result) -> None:
        self.make_result = make_result
        self.runs = 0

    def run(self, source: Any, clips: Sequence[ClipManifest]) -> VerificationReport:
        self.runs += 1
        ids = [item.clip.clip_id for item in source.load(clips)]
        results = tuple(
            self.make_result("R1", Verdict.PASS, 0.1, 1.0, 0.01, clip_id=c) for c in ids
        )
        return VerificationReport(results, (), tuple(clips))


def _candidate(store: CampaignStore) -> CandidateRecord:
    record = CandidateRecord(
        candidate_id=f"{PROMPT}__k00",
        prompt_id=PROMPT,
        split="train",
        k=0,
        seed=0,
        model_tag=BASE_TAG,
        video_path=str(store.video_path(BASE_TAG, PROMPT, 0)),
        latents_path=None,
        embeds_path=str(store.embeds_path(PROMPT)),
        config_hash="h",
    )
    write_record(store.meta_path(BASE_TAG, PROMPT, 0), record)
    return record


def _prompt() -> PromptRecord:
    return PromptRecord(
        prompt_id=PROMPT,
        text="a ball falls",
        queries=[QueryEntry(phrase="ball", expected_count=1)],
        family="fall",
        template_id="fall_drop",
        target_checks=["R1"],
        split="train",
    )


def test_a_score_sealed_by_the_previous_scorer_version_is_rescored(
    settings: Settings, make_result: Result
) -> None:
    write_artifact(
        default_artifact_path(settings),
        profile=settings.profile,
        checks_hash=active_checks_hash(settings),
    )
    store = CampaignStore.from_settings(settings)
    record = _candidate(store)
    scorer = CalibratedScorer(settings, checks())
    runner = _Runner(make_result)
    clips = [_clip(record.clip_id)]
    deps = ScoringDeps(lambda subset: clips, _Source(), runner, cache_guard(settings, False))
    score_candidates(store, BASE_TAG, [_prompt()], scorer, deps)
    path = store.score_path(BASE_TAG, scorer.name, record.candidate_id)
    old = read_record(path, CandidateScore).model_copy(update={"scorer_version": PREVIOUS_VERSION})
    write_record(path, old)
    assert stale_scores([old], scorer, [record]) == [record.candidate_id]
    (rescored,) = score_candidates(store, BASE_TAG, [_prompt()], scorer, deps)
    assert runner.runs == 2
    assert rescored.scorer_version == CalibratedScorer.version != PREVIOUS_VERSION
