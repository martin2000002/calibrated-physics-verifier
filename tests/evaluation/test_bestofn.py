from __future__ import annotations

from collections.abc import Callable

from physverify.evaluation.bestofn import pick_best
from physverify.finetune.records import CandidateScore
from physverify.finetune.settings import PairSettings

Score = Callable[..., CandidateScore]
RULES = PairSettings(min_margin=0.1, min_motion=0.5)


def test_the_best_eligible_candidate_wins(make_score: Score) -> None:
    candidates = {
        0: make_score("p__k00", 0.5),
        1: make_score("p__k01", 0.9, motion=0.0),
        2: make_score("p__k02", 0.7),
    }
    pick = pick_best(candidates, RULES)
    assert pick is not None
    assert (pick.k, pick.value, pick.n_eligible, pick.fallback) == (2, 0.7, 2, None)


def test_a_tie_goes_to_the_smaller_k(make_score: Score) -> None:
    pick = pick_best({3: make_score("p__k03", 0.6), 1: make_score("p__k01", 0.6)}, RULES)
    assert pick is not None and pick.k == 1


def test_fallbacks_are_named(make_score: Score) -> None:
    ineligible = pick_best(
        {0: make_score("p__k00", 0.4, objects_ok=False), 1: make_score("p__k01", 0.8, motion=0.0)},
        RULES,
    )
    assert ineligible is not None
    assert (ineligible.k, ineligible.fallback) == (1, "ineligible")
    abstained = pick_best({2: make_score("p__k02", None), 1: make_score("p__k01", None)}, RULES)
    assert abstained is not None
    assert (abstained.k, abstained.value, abstained.fallback) == (1, None, "all_abstained")
    assert pick_best({}, RULES) is None


def test_a_custom_key_selects(make_score: Score) -> None:
    candidates = {0: make_score("p__k00", 0.9), 1: make_score("p__k01", 0.1)}
    pick = pick_best(candidates, RULES, key=lambda s: None if s.score is None else -s.score)
    assert pick is not None and pick.k == 1
