from __future__ import annotations

from collections.abc import Callable

import pytest

from physverify.config.models import Settings
from physverify.finetune.records import CandidateScore
from physverify.finetune.scorers._base import AggregateScore
from physverify.finetune.selection import choose, evaluate_checkpoint, spaced
from physverify.finetune.settings import SelectSettings, load_finetune

Score = Callable[..., CandidateScore]


ON_R1 = {"base": 0.4, "lora-x": 0.9}


def rescore(score: CandidateScore, only: frozenset[str]) -> AggregateScore:
    """The stored score, or the ``ON_R1`` value when re-scored on R1 alone."""
    value = ON_R1[score.model_tag] if only == {"R1"} else score.score
    return AggregateScore(value, None, False, None, len(only), 0, 1.0, only)


@pytest.fixture
def cfg(settings: Settings) -> SelectSettings:
    return load_finetune(settings).select


@pytest.fixture
def base(make_score: Score) -> list[CandidateScore]:
    return [make_score(f"p__k0{k}", 0.5) for k in range(4)]


def _tagged(
    make_score: Score, values: list[float | None], **update: object
) -> list[CandidateScore]:
    return [make_score(f"p__k0{k}", v, model_tag="lora-x", **update) for k, v in enumerate(values)]


def test_spaced_keeps_the_last_checkpoint() -> None:
    steps = [100, 200, 300, 400, 500, 600]
    picked = spaced(steps, 3)
    assert len(picked) == 3 and picked[-1] == 600 and picked[0] == 100
    assert spaced([200, 100], 3) == [100, 200]


def test_the_paired_difference_uses_candidates_decided_in_both(
    make_score: Score, base: list[CandidateScore], cfg: SelectSettings
) -> None:
    scores = _tagged(make_score, [0.7, 0.6, 0.8, 0.9])
    result = evaluate_checkpoint(100, "a", "lora-x", scores, base, cfg, rescore)
    assert result.n_paired == 4
    assert result.paired_diff == pytest.approx(0.25)
    assert result.ci is not None and result.ci[0] < 0.25 < result.ci[1]
    assert not result.suspected_hacking


def test_the_paired_difference_is_taken_on_the_checks_decided_in_both(
    make_score: Score, cfg: SelectSettings
) -> None:
    base = [make_score("p__k00", 0.4, decided=["R1"])]
    tuned = [make_score("p__k00", 0.3, model_tag="lora-x", decided=["R1", "R2"])]
    result = evaluate_checkpoint(100, "a", "lora-x", tuned, base, cfg, rescore)
    assert result.n_paired == 1
    assert result.paired_diff == pytest.approx(0.5)
    disjoint = [make_score("p__k00", 0.9, model_tag="lora-x", decided=["R2"])]
    assert evaluate_checkpoint(100, "a", "t", disjoint, base, cfg, rescore).n_paired == 0


def test_more_abstention_or_less_motion_is_suspected_hacking(
    make_score: Score, base: list[CandidateScore], cfg: SelectSettings
) -> None:
    abstaining = _tagged(make_score, [0.7, 0.6, None, 0.8])
    result = evaluate_checkpoint(100, "a", "lora-x", abstaining, base, cfg, rescore)
    assert result.n_paired == 3 and result.abstention == 0.25
    assert result.suspected_hacking
    quiet = _tagged(make_score, [0.9] * 4, motion=1.0)
    assert evaluate_checkpoint(200, "b", "lora-y", quiet, base, cfg, rescore).suspected_hacking


def test_choose_skips_suspected_checkpoints(
    make_score: Score, base: list[CandidateScore], cfg: SelectSettings
) -> None:
    good = evaluate_checkpoint(100, "a", "t", _tagged(make_score, [0.6] * 4), base, cfg, rescore)
    better = evaluate_checkpoint(200, "b", "t", _tagged(make_score, [0.7] * 4), base, cfg, rescore)
    hacked = evaluate_checkpoint(
        300, "c", "t", _tagged(make_score, [0.99, None, None, None]), base, cfg, rescore
    )
    assert choose([good, better, hacked]) == better
    assert choose([hacked]) is None
