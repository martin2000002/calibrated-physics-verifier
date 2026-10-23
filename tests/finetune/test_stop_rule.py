"""The pilot's stop rule (item 1.6), pure part."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from physverify.finetune.records import CandidateScore, PairRecord, PairsReport
from physverify.finetune.settings import StopRuleSettings
from physverify.finetune.stop_rule import (
    PilotReview,
    StopVerdict,
    candidate_abstention,
    evaluate_stop,
    inspected_count,
    pairs_sha256,
    required_better,
    review_sample,
)

STAMP = ("2", "t")
RULES = StopRuleSettings()
FAMILIES = {"fall": ["R5a"]}


def _pair(i: int, **update: Any) -> PairRecord:
    fields: dict[str, Any] = {
        "scorer": "provisional",
        "scorer_version": STAMP[0],
        "calibrated": False,
        "thresholds": STAMP[1],
        "pair_id": f"p{i:02d}__k00__vs__p{i:02d}__k01",
        "prompt_id": f"p{i:02d}",
        "winner": f"p{i:02d}__k00",
        "loser": f"p{i:02d}__k01",
        "score_w": 0.9,
        "score_l": 0.5,
        "margin": 0.4,
        "weight": 1.0,
        "common_checks": ["R1"],
    }
    return PairRecord(**{**fields, **update})


def _report(pairs: int, scored: int, **update: Any) -> PairsReport:
    fields: dict[str, Any] = {
        "scorer": "provisional",
        "scorer_version": STAMP[0],
        "calibrated": False,
        "thresholds": STAMP[1],
        "prompts": scored,
        "scored_prompts": scored,
        "candidates": 4 * scored,
        "eligible": 4 * scored,
        "pairs": pairs,
        "dropped": {},
        "rejected": {},
        "yield_per_prompt": pairs / scored,
        "yield_per_scored_prompt": pairs / scored,
    }
    return PairsReport(**{**fields, **update})


def _review(pairs: list[PairRecord], better: int, rules: StopRuleSettings = RULES) -> PilotReview:
    return PilotReview(
        pairs_sha256=pairs_sha256(pairs),
        inspected=[p.pair_id for p in review_sample(pairs, rules)],
        better=better,
        created_utc="2026-10-19T00:00:00+00:00",
    )


def _verdict(
    pairs: list[PairRecord],
    report: PairsReport | None,
    review: PilotReview | None,
    abstention: float | None = 0.2,
    rules: StopRuleSettings = RULES,
) -> StopVerdict:
    return evaluate_stop(report, pairs, abstention, review, rules, FAMILIES, STAMP)


def _passing(n: int = 12, scored: int = 20) -> tuple[list[PairRecord], PairsReport]:
    return [_pair(i) for i in range(n)], _report(n, scored)


def test_a_good_pilot_passes() -> None:
    pairs, report = _passing()
    verdict = _verdict(pairs, report, _review(pairs, 6))
    assert verdict.passed and verdict.reasons == []
    assert verdict.families == FAMILIES and verdict.pairs_sha256 == pairs_sha256(pairs)
    assert verdict.measured["inspected"] == 10


def test_the_exact_edges_pass() -> None:
    pairs = [_pair(i) for i in range(6)]
    report = _report(6, 20)
    assert report.measured_yield == pytest.approx(0.3)
    assert _verdict(pairs, report, _review(pairs, 4), abstention=0.5).passed


def test_a_yield_below_the_edge_fires() -> None:
    pairs = [_pair(i) for i in range(5)]
    verdict = _verdict(pairs, _report(5, 20), _review(pairs, 5))
    assert not verdict.passed
    assert verdict.reasons == ["pairs per scored prompt 0.25 < 0.3"]


def test_too_much_abstention_fires() -> None:
    pairs, report = _passing()
    verdict = _verdict(pairs, report, _review(pairs, 6), abstention=0.51)
    assert verdict.reasons == ["candidate abstention 0.51 > 0.5"]


def test_no_scores_fire() -> None:
    pairs, report = _passing()
    assert _verdict(pairs, report, _review(pairs, 6), abstention=None).reasons == [
        "candidate abstention None > 0.5"
    ]


def test_a_missing_review_fires() -> None:
    pairs, report = _passing()
    assert _verdict(pairs, report, None).reasons == [
        "no pilot review recorded: run 'finetune pilot-review'"
    ]


def test_a_review_of_other_pairs_does_not_count() -> None:
    pairs, report = _passing()
    review = _review(pairs, 10)
    changed = [*pairs[:-1], _pair(11, margin=0.3)]
    verdict = _verdict(changed, report, review)
    assert verdict.reasons == ["the pilot review was made on other pairs: review the current ones"]


def test_a_review_of_another_sample_does_not_count() -> None:
    pairs, report = _passing()
    review = _review(pairs, 6).model_copy(update={"inspected": [p.pair_id for p in pairs[:10]]})
    drawn = {p.pair_id for p in review_sample(pairs, RULES)}
    assert drawn != set(review.inspected)
    assert "not the drawn sample" in _verdict(pairs, report, review).reasons[0]


def test_too_few_better_winners_fire() -> None:
    pairs, report = _passing()
    verdict = _verdict(pairs, report, _review(pairs, 5))
    assert verdict.reasons == ["5 of 10 inspected pairs show a better winner (< 6)"]


def test_six_pairs_are_all_inspected_and_need_four() -> None:
    pairs = [_pair(i) for i in range(6)]
    assert inspected_count(6, RULES) == 6 and required_better(6, RULES) == 4
    assert len(review_sample(pairs, RULES)) == 6
    report = _report(6, 20)
    assert _verdict(pairs, report, _review(pairs, 4)).passed
    assert not _verdict(pairs, report, _review(pairs, 3)).passed


def test_pairs_of_another_scorer_stamp_fire() -> None:
    pairs = [_pair(i, thresholds="old") for i in range(12)]
    verdict = _verdict(pairs, _report(12, 20), _review(pairs, 6))
    assert verdict.reasons == [
        "the pairs were built under another scorer stamp: run 'finetune pairs' again"
    ]


def test_no_pairs_at_all_fire_on_every_count() -> None:
    verdict = _verdict([], None, None, abstention=None)
    assert len(verdict.reasons) == 3 and not verdict.passed


def test_the_sample_is_seeded() -> None:
    pairs = [_pair(i) for i in range(30)]
    first = review_sample(pairs, RULES)
    assert first == review_sample(pairs, RULES)
    assert first != review_sample(pairs, RULES.model_copy(update={"seed": 1}))


def test_candidate_abstention(make_score: Callable[..., CandidateScore]) -> None:
    scores = [make_score("p__k00", None), make_score("p__k01", 0.4)]
    assert candidate_abstention(scores) == 0.5
    assert candidate_abstention([]) is None


def test_enforcement_is_recorded() -> None:
    rules = RULES.model_copy(update={"enforce": False})
    verdict = _verdict([], None, None, abstention=None, rules=rules)
    assert not verdict.passed and not verdict.enforced
