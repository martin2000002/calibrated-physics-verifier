from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from physverify.calibration.apply import ClipScore
from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.datasets.local_video import QueryEntry
from physverify.finetune.records import (
    CandidateRecord,
    CandidateScore,
    PromptRecord,
    read_record,
    write_record,
)
from physverify.finetune.scorers.calibrated import CalibratedScorer, to_aggregate
from physverify.finetune.scorers.provisional import ProvisionalScorer, provisional_score
from physverify.finetune.scoring import (
    ScoringDeps,
    pending,
    rescorer,
    score_candidates,
    scorer_registry,
    stale_scores,
)
from physverify.finetune.seeds import WHOLE
from physverify.finetune.store import CampaignStore
from physverify.schema.checks import CheckResult, Verdict
from physverify.schema.clip import ClipManifest, Split, TimeSource
from physverify.schema.states import ObjectStateTable
from physverify.verification.inputs import ClipInputs
from physverify.verification.runner import RowStatus, TableRow, VerificationReport

Result = Callable[..., CheckResult]
PROMPT = "fall-0001"


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


def _row(clip_id: str, check_id: str, magnitude: float, verdict: Verdict) -> TableRow:
    return TableRow(
        clip_id=clip_id,
        check_id=check_id,
        status=RowStatus.SCORED,
        magnitude=magnitude,
        sigma=math.nan,
        verdict=verdict,
        abstain_reason=None,
        n_subjects=1,
        n_abstained=0,
        unit="",
        row_step=0,
    )


class FakeSource:
    def __init__(self, states: ObjectStateTable) -> None:
        self.states = states

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        for clip in clips:
            yield ClipInputs(clip, SimpleNamespace(states=self.states), None)  # type: ignore[arg-type]


class FakeRunner:
    """Fixed verdicts: R1 PASS at 0.2 of its threshold, R2 FAIL at 1.5 of it."""

    def __init__(self, make_result: Result) -> None:
        self.make_result = make_result
        self.runs = 0

    def run(self, source: Any, clips: Sequence[ClipManifest]) -> VerificationReport:
        self.runs += 1
        items = list(source.load(clips))
        results, rows = [], []
        for item in items:
            cid = item.clip.clip_id
            results += [
                self.make_result("R1", Verdict.PASS, 0.2, 1.0, clip_id=cid),
                self.make_result("R2", Verdict.FAIL, 3.0, 2.0, clip_id=cid),
            ]
            rows += [_row(cid, "R1", 0.2, Verdict.PASS), _row(cid, "R2", 3.0, Verdict.FAIL)]
        return VerificationReport(tuple(results), tuple(rows), tuple(clips))


def _write_candidates(store: CampaignStore, n: int) -> list[CandidateRecord]:
    records = []
    for k in range(n):
        record = CandidateRecord(
            candidate_id=f"{PROMPT}__k{k:02d}",
            prompt_id=PROMPT,
            split="train",
            k=k,
            seed=k,
            model_tag="base",
            video_path=str(store.video_path("base", PROMPT, k)),
            latents_path=None,
            embeds_path=str(store.embeds_path(PROMPT)),
            config_hash="h",
        )
        write_record(store.meta_path("base", PROMPT, k), record)
        records.append(record)
    return records


@pytest.fixture
def prompt() -> PromptRecord:
    return PromptRecord(
        prompt_id=PROMPT,
        text="a ball falls",
        queries=[QueryEntry(phrase="ball", expected_count=1)],
        family="fall",
        template_id="fall_drop",
        target_checks=["R5a"],
        split="train",
    )


@pytest.fixture
def states(make_states: Callable[..., ObjectStateTable]) -> ObjectStateTable:
    position = np.zeros((4, 1, 3))
    position[:, 0, 1] = [0.0, 1.0, 2.0, 3.0]
    return make_states(position, np.ones((4, 1, 3)), ["ball"])


def test_both_scorers_are_registered() -> None:
    assert sorted(scorer_registry()) == ["calibrated", "provisional"]


def test_scoring_writes_stamped_scores_and_resumes(
    tmp_path: Path,
    settings: Settings,
    prompt: PromptRecord,
    states: ObjectStateTable,
    make_result: Result,
) -> None:
    store = CampaignStore(tmp_path / "ft", tmp_path / "raw", "c")
    _write_candidates(store, 3)
    runner = FakeRunner(make_result)
    deps = ScoringDeps(
        clips=lambda subset: [_clip(r.clip_id) for r in store.candidates("base")],
        source=FakeSource(states),
        runner=runner,
    )
    scorer = ProvisionalScorer(settings)
    scores = score_candidates(store, "base", [prompt], scorer, deps)
    assert len(scores) == 3 and runner.runs == 1
    first = scores[0]
    assert (first.scorer, first.calibrated) == ("provisional", False)
    assert first.thresholds == scorer.thresholds()
    assert first.score == pytest.approx(1 / math.sqrt(1.2 * 2.5))
    assert (first.n_decided, first.n_failed, first.decided) == (2, 1, ["R1", "R2"])
    assert first.per_check["R2"].verdict == "FAIL"
    assert first.motion == pytest.approx(3.0)
    assert first.objects_ok is True
    assert Path(first.results_path).is_file()
    assert store.scores_path("base", "provisional").is_file()
    score_candidates(store, "base", [prompt], scorer, deps)
    assert runner.runs == 1
    assert rescorer(scorer)(first, frozenset({"R1"})).score == pytest.approx(1 / 1.2)


def _scored_store(
    tmp_path: Path, settings: Settings, prompt: PromptRecord, states: ObjectStateTable, runner: Any
) -> tuple[CampaignStore, ScoringDeps, ProvisionalScorer]:
    store = CampaignStore(tmp_path / "ft", tmp_path / "raw", "c")
    _write_candidates(store, 3)
    deps = ScoringDeps(
        clips=lambda subset: [_clip(r.clip_id) for r in store.candidates("base")],
        source=FakeSource(states),
        runner=runner,
    )
    scorer = ProvisionalScorer(settings)
    score_candidates(store, "base", [prompt], scorer, deps)
    return store, deps, scorer


def _rewrite(store: CampaignStore, cid: str, **update: Any) -> None:
    path = store.score_path("base", "provisional", cid)
    write_record(path, read_record(path, CandidateScore).model_copy(update=update))


def test_scores_carry_the_candidate_config_hash(
    tmp_path: Path,
    settings: Settings,
    prompt: PromptRecord,
    states: ObjectStateTable,
    make_result: Result,
) -> None:
    store, _, scorer = _scored_store(tmp_path, settings, prompt, states, FakeRunner(make_result))
    scores = store.scores("base", "provisional")
    assert {s.candidate_config_hash for s in scores} == {"h"}
    assert stale_scores(scores, scorer, store.candidates("base")) == []


def test_a_stale_stamp_or_candidate_hash_is_scored_again(
    tmp_path: Path,
    settings: Settings,
    prompt: PromptRecord,
    states: ObjectStateTable,
    make_result: Result,
) -> None:
    runner = FakeRunner(make_result)
    store, deps, scorer = _scored_store(tmp_path, settings, prompt, states, runner)
    _rewrite(store, f"{PROMPT}__k00", scorer_version="0")
    _rewrite(store, f"{PROMPT}__k01", candidate_config_hash=None)
    kept = store.score_path("base", "provisional", f"{PROMPT}__k02").read_bytes()
    stale = stale_scores(store.scores("base", "provisional"), scorer, store.candidates("base"))
    assert stale == [f"{PROMPT}__k00", f"{PROMPT}__k01"]
    todo = pending(store, "base", scorer, {PROMPT: prompt}, WHOLE)
    assert [r.candidate_id for r in todo] == stale
    scores = score_candidates(store, "base", [prompt], scorer, deps)
    assert runner.runs == 2
    assert stale_scores(scores, scorer, store.candidates("base")) == []
    assert store.score_path("base", "provisional", f"{PROMPT}__k02").read_bytes() == kept


def test_a_candidate_with_another_config_hash_is_pending(
    tmp_path: Path,
    settings: Settings,
    prompt: PromptRecord,
    states: ObjectStateTable,
    make_result: Result,
) -> None:
    store, _, scorer = _scored_store(tmp_path, settings, prompt, states, FakeRunner(make_result))
    meta = store.meta_path("base", PROMPT, 1)
    write_record(meta, read_record(meta, CandidateRecord).model_copy(update={"config_hash": "h2"}))
    todo = pending(store, "base", scorer, {PROMPT: prompt}, WHOLE)
    assert [r.candidate_id for r in todo] == [f"{PROMPT}__k01"]


def test_provisional_score_by_hand(make_result: Result) -> None:
    results = [
        make_result("R1", Verdict.PASS, 0.25, 1.0, sigma=0.1),
        make_result("R2", Verdict.PASS, 0.5, 2.0),
        make_result("R3a", Verdict.ABSTAIN, None, None),
    ]
    score = provisional_score(results)
    assert score.score == pytest.approx(0.8)
    assert score.sigma is None
    assert score.coverage == pytest.approx(2 / 3)
    assert score.decided == frozenset({"R1", "R2"})
    assert dict(score.per_check) == pytest.approx({"R1": 0.8, "R2": 0.8})
    only = provisional_score(results, frozenset({"R3a"}))
    assert only.score is None and only.abstained


def test_provisional_sigma_by_hand(make_result: Result) -> None:
    results = [
        make_result("R1", Verdict.PASS, 0.25, 1.0, sigma=0.1),
        make_result("R2", Verdict.PASS, 1.0, 2.0, sigma=0.4),
    ]
    score = provisional_score(results)
    expected = 1 / math.sqrt(1.25 * 1.5)
    assert score.score == pytest.approx(expected)
    assert score.sigma == pytest.approx(expected / 2 * math.hypot(0.1 / 1.25, 0.2 / 1.5))


def test_a_clamped_term_adds_no_sigma(make_result: Result) -> None:
    results = [
        make_result("R1", Verdict.PASS, -0.5, 1.0),
        make_result("R2", Verdict.PASS, 1.0, 2.0, sigma=0.4),
    ]
    score = provisional_score(results)
    expected = 1 / math.sqrt(1.5)
    assert score.score == pytest.approx(expected)
    assert score.sigma == pytest.approx(expected / 2 * (0.2 / 1.5))


@pytest.mark.parametrize("ratio", [0.0, 0.3, 1.0, 4.0, 100.0])
def test_one_check_scores_one_over_one_plus_rho(make_result: Result, ratio: float) -> None:
    verdict = Verdict.PASS if ratio <= 1.0 else Verdict.FAIL
    score = provisional_score([make_result("R1", verdict, ratio * 0.05, 0.05)])
    assert score.score == pytest.approx(1 / (1 + ratio))


@pytest.mark.parametrize(("low", "high"), [(0.0, 0.5), (0.9, 1.1), (2.0, 2.01), (12.7, 49.0)])
def test_the_score_falls_strictly_with_each_ratio(
    make_result: Result, low: float, high: float
) -> None:
    def scored(ratio: float) -> float:
        rows = [
            make_result("R1", Verdict.FAIL, 2.0, 1.0),
            make_result("R5a", Verdict.FAIL, ratio * 0.15, 0.15),
        ]
        value = provisional_score(rows).score
        assert value is not None
        return value

    assert scored(low) > scored(high)


def test_pass_and_fail_sit_on_either_side_of_one_half(make_result: Result) -> None:
    at_threshold = [
        make_result("R1", Verdict.PASS, 0.05, 0.05, sigma=0.0),
        make_result("R2", Verdict.PASS, 0.01, 0.01, sigma=0.0),
    ]
    assert provisional_score(at_threshold).score == pytest.approx(0.5)
    passing = provisional_score([make_result("R1", Verdict.PASS, 0.01, 0.05)])
    assert passing.score is not None and passing.score >= 0.5
    failing = provisional_score([make_result("R2", Verdict.FAIL, 1.0, 0.01)])
    assert failing.per_check["R2"] < 0.5 and failing.n_failed == 1


def test_a_huge_violation_still_scores_above_zero(make_result: Result) -> None:
    score = provisional_score([make_result("R1", Verdict.FAIL, 1e6, 1.0)])
    assert score.score is not None and score.score > 0.0


def test_a_negative_magnitude_counts_as_zero(make_result: Result) -> None:
    score = provisional_score([make_result("R1", Verdict.PASS, -3.0, 1.0)])
    assert score.score == pytest.approx(1.0)


def test_the_provisional_scorer_is_version_two() -> None:
    assert ProvisionalScorer.version == "2"


def test_a_missing_calibration_module_makes_the_scorer_unavailable(
    tmp_path: Path, settings: Settings, prompt: PromptRecord, make_result: Result
) -> None:
    def missing(name: str) -> Any:
        raise ImportError(f"No module named {name!r}")

    scorer = CalibratedScorer(settings, import_module=missing)
    assert scorer.unavailable() is not None
    store = CampaignStore(tmp_path / "ft", tmp_path / "raw", "c")
    deps = ScoringDeps(lambda s: [], FakeSource(None), FakeRunner(make_result))  # type: ignore[arg-type]
    with pytest.raises(UsageError, match="unavailable"):
        score_candidates(store, "base", [prompt], scorer, deps)


def _clip_score(**update: Any) -> ClipScore:
    fields: dict[str, Any] = {
        "clip_id": "c",
        "score": 0.6,
        "sigma": 0.05,
        "coverage": 0.8,
        "abstained": False,
        "reason": None,
        "n_failed": 1,
        "decided_checks": ("R1", "R2"),
        "abstained_checks": ("R3a",),
        "per_check": {"R1": 0.9, "R2": 0.4},
    }
    return ClipScore(**{**fields, **update})


def test_the_calibrated_scorer_wraps_score_clip(settings: Settings, make_result: Result) -> None:
    calls: list[tuple[Any, ...]] = []
    stratum = _stratum()

    def score_clip(results: Any, got: Any, k: float | None = None, only: Any = None) -> ClipScore:
        calls.append((got, k, only))
        return _clip_score()

    scorer = _scorer(settings, _calibration_modules(score_clip, stratum))
    assert scorer.unavailable() is None
    assert scorer.thresholds() == (
        f"calibrated:thresholds.json#perceived/precise@run7@checks:{HASH[:12]}"
    )
    only = frozenset({"R1"})
    score = scorer.score([make_result("R1", Verdict.PASS)], only=only)
    assert calls == [(stratum, None, only)]
    assert score.score == 0.6 and score.sigma == 0.05 and score.coverage == 0.8
    assert (score.n_decided, score.n_failed) == (2, 1)
    assert score.decided == frozenset({"R1", "R2"})
    assert dict(score.per_check) == {"R1": 0.9, "R2": 0.4}


HASH = "a" * 64


class FakeCalibrationError(Exception):
    pass


def _stratum() -> SimpleNamespace:
    return SimpleNamespace(
        source="thresholds.json#perceived/precise@run7",
        time_base_problem=lambda profile, step: None,
        check_status=lambda: {"source": "thresholds.json#perceived/precise@run7"},
    )


def _scorer(settings: Settings, modules: dict[str, Any]) -> CalibratedScorer:
    def active_checks_hash(got: Settings, registry: Any) -> str:
        assert got is settings and registry == {"R1": object}
        return HASH

    modules["physverify.calibration.settings"] = SimpleNamespace(
        active_checks_hash=active_checks_hash,
        calibration_settings=lambda s: SimpleNamespace(require_calibrated_artifact=False),
    )
    return CalibratedScorer(
        settings,
        import_module=modules.__getitem__,  # type: ignore[arg-type]
        checks=lambda: {"R1": object},  # type: ignore[dict-item]
    )


def _calibration_modules(score_clip: Any, stratum: Any) -> dict[str, Any]:
    def open_stratum(path: Path, level: Any, profile: str, **options: Any) -> Any:
        assert options["checks_hash"] == HASH and not options["require_calibrated"]
        return stratum

    return {
        "physverify.calibration.apply": SimpleNamespace(score_clip=score_clip),
        "physverify.calibration.loader": SimpleNamespace(
            default_artifact_path=lambda s: Path("cal/thresholds.json"),
            open_stratum=open_stratum,
            unique_row_step=lambda steps: max(steps),
            CalibrationError=FakeCalibrationError,
        ),
    }


def test_all_zero_weights_map_to_an_abstention() -> None:
    clip = _clip_score(
        score=None, sigma=None, abstained=True, reason="no_informative_check", decided_checks=()
    )
    score = to_aggregate(clip)
    assert score.score is None and score.abstained
    assert score.reason == "no_informative_check"
    assert score.n_decided == 0


def test_a_nan_subject_listed_first_does_not_set_the_term(make_result: Result) -> None:
    results = [
        make_result("R1", Verdict.PASS, math.nan, 1.0),
        make_result("R1", Verdict.PASS, 0.5, 1.0),
    ]
    assert dict(provisional_score(results).per_check) == pytest.approx({"R1": 1 / 1.5})
