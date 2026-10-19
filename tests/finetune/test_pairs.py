from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from physverify.config.models import Settings
from physverify.finetune.pairs import (
    ABSTAINED,
    LOW_MOTION,
    MARGIN,
    NO_COMMON,
    OBJECTS,
    QUIET_WINNER,
    SIGMA,
    SINGLE,
    TIE,
    build_pairs,
    form_pairs,
    pair_weight,
    stamp_of,
)
from physverify.finetune.records import CandidateScore, PromptRecord
from physverify.finetune.scorers._base import AggregateScore
from physverify.finetune.scorers.provisional import provisional_score
from physverify.finetune.settings import PairSettings, load_finetune
from physverify.schema.checks import CheckResult, Verdict

Score = Callable[..., CandidateScore]
Result = Callable[..., CheckResult]
Rows = Sequence[tuple[str, Verdict, float, float, float]]
Key = tuple[str, frozenset[str]]


class FakeRescore:
    """The stored score unless ``table`` overrides it for (candidate, checks)."""

    def __init__(self, table: Mapping[Key, float] | None = None) -> None:
        self.table = dict(table or {})
        self.calls: list[Key] = []

    def __call__(self, score: CandidateScore, only: frozenset[str]) -> AggregateScore:
        self.calls.append((score.candidate_id, only))
        value = self.table.get((score.candidate_id, only), score.score)
        return AggregateScore(value, score.sigma, False, None, len(only), 0, 1.0, only)


@pytest.fixture
def rules(settings: Settings) -> PairSettings:
    return load_finetune(settings).pairs


def _form(scores: list[CandidateScore], rules: PairSettings, rescore: Any = None) -> Any:
    return form_pairs(scores, rules, rescore or FakeRescore(), stamp_of(scores))


def test_a_tie_is_rejected(make_score: Score, rules: PairSettings) -> None:
    result = _form([make_score("p__k00", 0.7), make_score("p__k01", 0.7)], rules)
    assert result.pairs == [] and result.rejected == {TIE: 1}


def test_the_margin_rule_at_its_edge(make_score: Score, rules: PairSettings) -> None:
    below = _form([make_score("p__k00", 0.80), make_score("p__k01", 0.71)], rules)
    assert below.rejected == {MARGIN: 1}
    above = _form([make_score("p__k00", 0.80), make_score("p__k01", 0.69)], rules)
    assert [p.winner for p in above.pairs] == ["p__k00"]
    assert above.pairs[0].margin == pytest.approx(0.11)


def test_the_sigma_rule_needs_both_sigmas(make_score: Score, rules: PairSettings) -> None:
    noisy = [make_score("p__k00", 0.9, sigma=0.15), make_score("p__k01", 0.7, sigma=0.15)]
    assert _form(noisy, rules).rejected == {SIGMA: 1}
    one = [make_score("p__k00", 0.9, sigma=0.15), make_score("p__k01", 0.7)]
    assert len(_form(one, rules).pairs) == 1


def test_an_abstained_candidate_never_wins(make_score: Score, rules: PairSettings) -> None:
    scores = [
        make_score("p__k00", None, abstained=True),
        make_score("p__k01", 0.9),
        make_score("p__k02", 0.3),
    ]
    result = _form(scores, rules)
    assert result.dropped == {ABSTAINED: 1}
    assert [(p.winner, p.loser) for p in result.pairs] == [("p__k01", "p__k02")]


def test_a_quiet_winner_is_rejected(make_score: Score, rules: PairSettings) -> None:
    scores = [make_score("p__k00", 0.9, motion=0.6), make_score("p__k01", 0.3, motion=2.0)]
    assert _form(scores, rules).rejected == {QUIET_WINNER: 1}


def test_wrong_object_count_and_low_motion_drop_candidates(
    make_score: Score, rules: PairSettings
) -> None:
    scores = [
        make_score("p__k00", 0.9, objects_ok=False),
        make_score("p__k01", 0.8, motion=0.1),
        make_score("p__k02", 0.3),
    ]
    result = _form(scores, rules)
    assert result.dropped == {OBJECTS: 1, LOW_MOTION: 1}
    assert result.rejected == {SINGLE: 1}
    relaxed = rules.model_copy(update={"require_object_count": False})
    assert _form(scores, relaxed).pairs[0].winner == "p__k00"


def test_several_pairs_go_best_against_worst(make_score: Score, rules: PairSettings) -> None:
    scores = [make_score(f"p__k0{i}", v) for i, v in enumerate([0.5, 0.9, 0.1, 0.7])]
    result = _form(scores, rules.model_copy(update={"max_pairs_per_prompt": 2}))
    assert [(p.winner, p.loser) for p in result.pairs] == [
        ("p__k01", "p__k02"),
        ("p__k03", "p__k00"),
    ]


def test_no_common_decided_check_gives_no_pair(make_score: Score, rules: PairSettings) -> None:
    scores = [make_score("p__k00", 0.9, decided=["R1"]), make_score("p__k01", 0.2, decided=["R2"])]
    assert _form(scores, rules).rejected == {NO_COMMON: 1}


def test_pairs_are_decided_on_the_common_checks(make_score: Score, rules: PairSettings) -> None:
    """A wins on its stored score only through R2, which B abstained on; on R1 alone B wins."""
    a = make_score("p__k00", 0.9, decided=["R1", "R2"])
    b = make_score("p__k01", 0.5, decided=["R1"])
    common = frozenset({"R1"})
    rescore = FakeRescore({("p__k00", common): 0.3, ("p__k01", common): 0.5})
    result = _form([a, b], rules, rescore)
    assert set(rescore.calls) == {("p__k00", common), ("p__k01", common)}
    pair = result.pairs[0]
    assert (pair.winner, pair.loser) == ("p__k01", "p__k00")
    assert (pair.score_w, pair.score_l, pair.common_checks) == (0.5, 0.3, ["R1"])


def _prompts(*ids: str) -> list[PromptRecord]:
    return [
        PromptRecord(
            prompt_id=pid,
            text=pid,
            queries=["ball"],
            family="fall",
            template_id="fall_drop",
            target_checks=["R5a"],
            split="train",
        )
        for pid in ids
    ]


def test_the_report_accounts_for_every_candidate(make_score: Score, rules: PairSettings) -> None:
    prompts = _prompts("p", "q")
    scores = [
        make_score("p__k00", 0.9),
        make_score("p__k01", 0.2),
        make_score("p__k02", None, abstained=True),
        make_score("q__k00", 0.5, motion=0.0),
        make_score("q__k01", 0.4),
    ]
    pairs, report = build_pairs(prompts, scores, rules, FakeRescore())
    assert report.candidates == len(scores)
    assert report.eligible + sum(report.dropped.values()) == report.candidates
    assert report.pairs == len(pairs) == 1
    assert report.yield_per_prompt == report.yield_per_scored_prompt == 0.5
    assert report.scorer == "provisional" and not report.calibrated


def test_unscored_prompts_lower_only_the_all_prompt_yield(
    make_score: Score, rules: PairSettings
) -> None:
    scores = [make_score("p__k00", 0.9), make_score("p__k01", 0.2)]
    _, report = build_pairs(_prompts("p", "q", "r", "s"), scores, rules, FakeRescore())
    assert (report.pairs, report.prompts, report.scored_prompts) == (1, 4, 1)
    assert report.yield_per_prompt == 0.25
    assert report.yield_per_scored_prompt == 1.0


def test_mixed_scorers_are_refused(make_score: Score) -> None:
    with pytest.raises(ValueError, match="mix"):
        stamp_of([make_score("p__k00", 0.5), make_score("p__k01", 0.4, thresholds="other")])


def test_pair_weight(rules: PairSettings) -> None:
    assert pair_weight(0.1, rules) == 1.0
    weighted = rules.model_copy(update={"weight_by_margin": True, "margin_ref": 0.5})
    assert pair_weight(0.1, weighted) == pytest.approx(0.2)
    assert pair_weight(0.9, weighted) == 1.0


class ProvisionalRescore:
    """The real provisional scorer over in-memory check results, keyed by candidate."""

    def __init__(self, results: Mapping[str, list[CheckResult]]) -> None:
        self.results = results

    def __call__(self, score: CandidateScore, only: frozenset[str]) -> AggregateScore:
        return provisional_score(self.results[score.candidate_id], only)


def _scored(
    make_score: Score, make_result: Result, candidates: Mapping[str, Rows]
) -> tuple[list[CandidateScore], ProvisionalRescore]:
    results = {
        cid: [make_result(c, v, m, t, sigma=s, clip_id=cid) for c, v, m, t, s in rows]
        for cid, rows in candidates.items()
    }
    scores = []
    for cid, rows in results.items():
        agg = provisional_score(rows)
        scores.append(
            make_score(
                cid,
                agg.score,
                sigma=agg.sigma,
                n_decided=agg.n_decided,
                n_failed=agg.n_failed,
                decided=sorted(agg.decided),
                scorer_version="2",
            )
        )
    return scores, ProvisionalRescore(results)


def _open(rules: PairSettings) -> PairSettings:
    return rules.model_copy(update={"min_margin": 0.0, "sigma_k": 0.0, "min_motion_ratio": 0.0})


FAIL = Verdict.FAIL
NAN = math.nan


def test_failing_candidates_rank_by_how_badly_they_fail(
    make_score: Score, make_result: Result, rules: PairSettings
) -> None:
    """The H200 smoke magnitudes: provisional v1 clipped every FAIL to 0 and tied this pair."""
    shared = [("R1", FAIL, 2.45, 0.05, NAN), ("R2", FAIL, 1.0, 0.01, NAN)]
    scores, rescore = _scored(
        make_score,
        make_result,
        {
            "p__k00": [*shared, ("R5a", FAIL, 1.9, 0.15, NAN)],
            "p__k01": [*shared, ("R5a", FAIL, 1.2, 0.15, NAN)],
        },
    )
    assert all(s.score is not None and 0.0 < s.score < 0.5 for s in scores)
    result = _form(scores, _open(rules), rescore)
    assert result.rejected == {}
    assert [(p.winner, p.loser) for p in result.pairs] == [("p__k01", "p__k00")]
    assert result.pairs[0].margin > 0.0


def test_a_shared_saturated_check_does_not_decide_the_pair(
    make_score: Score, make_result: Result, rules: PairSettings
) -> None:
    """R2 at its own ceiling (rho = 100) in both: the smaller R5a violation still wins."""
    saturated = ("R2", FAIL, 1.0, 0.01, NAN)
    scores, rescore = _scored(
        make_score,
        make_result,
        {
            "p__k00": [saturated, ("R5a", Verdict.PASS, 0.1, 0.15, NAN)],
            "p__k01": [saturated, ("R5a", FAIL, 1.9, 0.15, NAN)],
        },
    )
    result = _form(scores, _open(rules), rescore)
    assert result.rejected == {}
    assert [p.winner for p in result.pairs] == ["p__k00"]


def test_the_sigma_rule_holds_on_the_rescored_provisional_sigma(
    make_score: Score, make_result: Result, rules: PairSettings
) -> None:
    scores, rescore = _scored(
        make_score,
        make_result,
        {
            "p__k00": [("R5a", FAIL, 0.30, 0.15, 0.15)],
            "p__k01": [("R5a", FAIL, 0.33, 0.15, 0.15)],
        },
    )
    noisy = _open(rules).model_copy(update={"sigma_k": 2.0})
    assert _form(scores, noisy, rescore).rejected == {SIGMA: 1}
    assert len(_form(scores, _open(rules), rescore).pairs) == 1
