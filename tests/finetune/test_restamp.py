from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar

from physverify.config.models import Settings
from physverify.finetune.records import CandidateScore, CheckSummary
from physverify.finetune.scorers._base import AggregateScore
from physverify.finetune.scoring import restamp
from physverify.schema.checks import CheckResult


class NewScorer:
    name: ClassVar[str] = "calibrated"
    version: ClassVar[str] = "3"
    calibrated: ClassVar[bool] = True

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    def unavailable(self) -> str | None:
        return None

    def thresholds(self) -> str:
        return "calibrated:kubric@checks:abc"

    def score(
        self, results: Sequence[CheckResult], only: frozenset[str] | None = None
    ) -> AggregateScore:
        raise AssertionError("restamp takes the aggregate, it does not score")


def test_restamp_replaces_stamp_and_aggregate_and_keeps_signals(
    make_score: Callable[..., CandidateScore],
) -> None:
    summary = CheckSummary(status="scored", magnitude=0.1, sigma=None, verdict="pass", unit="")
    old = make_score("p__k00", 0.4, per_check={"R1": summary}, results_path="r.jsonl")
    aggregate = AggregateScore(
        score=0.8,
        sigma=0.05,
        abstained=False,
        reason=None,
        n_decided=1,
        n_failed=0,
        coverage=0.5,
        decided=frozenset({"R2", "R1"}),
    )
    new = restamp(old, NewScorer(), aggregate)
    assert (new.scorer, new.scorer_version, new.calibrated) == ("calibrated", "3", True)
    assert new.thresholds == "calibrated:kubric@checks:abc"
    assert (new.score, new.sigma, new.coverage, new.decided) == (0.8, 0.05, 0.5, ["R1", "R2"])
    assert new.per_check == old.per_check
    assert (new.motion, new.objects_ok, new.results_path) == (2.0, True, "r.jsonl")
    assert old.score == 0.4


def test_restamp_of_a_none_score_abstains(make_score: Callable[..., CandidateScore]) -> None:
    aggregate = AggregateScore(None, None, False, "no_checks", 0, 0, 0.0)
    assert restamp(make_score("p__k00", 0.4), NewScorer(), aggregate).abstained
