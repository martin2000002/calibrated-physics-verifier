"""Winner-loser pairs per prompt.

Pure: the stored scores and a ``rescore`` function come in, pairs and counts go out. A pair is
decided on the score recomputed over the checks decided in *both* candidates, so a candidate
cannot win by abstaining on the checks it would fail.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from physverify.finetune.records import (
    CandidateScore,
    PairRecord,
    PairsReport,
    PromptRecord,
    ScorerStamp,
    by_prompt,
)
from physverify.finetune.scorers._base import AggregateScore
from physverify.finetune.settings import PairSettings

Rescore = Callable[[CandidateScore, frozenset[str]], AggregateScore]

ABSTAINED = "abstained"
NO_SCORE = "no_score"
NO_DECIDED = "no_decided"
OBJECTS = "objects"
LOW_MOTION = "low_motion"
LOW_COVERAGE = "low_coverage"
NO_COMMON = "no_common_checks"
TIE = "tie"
MARGIN = "margin"
SIGMA = "sigma"
QUIET_WINNER = "quiet_winner"
SINGLE = "single_eligible"


def drop_reason(score: CandidateScore, rules: PairSettings) -> str | None:
    """Why a candidate leaves its prompt, or None if it is eligible (rule 1)."""
    if score.abstained:
        return ABSTAINED
    if score.score is None:
        return NO_SCORE
    if score.n_decided < 1:
        return NO_DECIDED
    if rules.require_object_count and score.objects_ok is False:
        return OBJECTS
    if score.motion is None or score.motion < rules.min_motion:
        return LOW_MOTION
    if score.coverage < rules.min_coverage:
        return LOW_COVERAGE
    return None


@dataclass(frozen=True)
class Judgement:
    """A judged pair: winner first, or the reason it was rejected."""

    winner: CandidateScore
    loser: CandidateScore
    score_w: float
    score_l: float
    common: frozenset[str]
    rejected: str | None


def _ranked(scores: Sequence[CandidateScore]) -> list[CandidateScore]:
    return sorted(scores, key=lambda s: (-(s.score or 0.0), s.candidate_id))


@dataclass(frozen=True)
class CommonScores:
    """Two candidates re-scored on the checks decided in both."""

    a: AggregateScore
    b: AggregateScore
    score_a: float
    score_b: float
    common: frozenset[str]


def on_common_checks(
    first: CandidateScore, second: CandidateScore, rescore: Rescore
) -> CommonScores | None:
    """None when no check is decided in both, or one re-scored side abstains."""
    common = frozenset(first.decided) & frozenset(second.decided)
    if not common:
        return None
    a, b = rescore(first, common), rescore(second, common)
    if a.score is None or b.score is None:
        return None
    return CommonScores(a, b, a.score, b.score, common)


def judge(
    high: CandidateScore, low: CandidateScore, rules: PairSettings, rescore: Rescore
) -> Judgement:
    both = on_common_checks(high, low, rescore)
    if both is None:
        common = frozenset(high.decided) & frozenset(low.decided)
        return Judgement(high, low, math.nan, math.nan, common, NO_COMMON)
    a, b, score_a, score_b, common = both.a, both.b, both.score_a, both.score_b, both.common
    if score_b > score_a:
        high, low, a, b, score_a, score_b = low, high, b, a, score_b, score_a
    return Judgement(high, low, score_a, score_b, common, _rejection(a, b, high, low, rules))


def _rejection(
    a: AggregateScore,
    b: AggregateScore,
    winner: CandidateScore,
    loser: CandidateScore,
    rules: PairSettings,
) -> str | None:
    assert a.score is not None and b.score is not None
    delta = a.score - b.score
    if delta == 0:
        return TIE
    if delta < rules.min_margin:
        return MARGIN
    both = a.sigma is not None and b.sigma is not None
    if both and delta <= rules.sigma_k * math.hypot(a.sigma or 0.0, b.sigma or 0.0):
        return SIGMA
    if (winner.motion or 0.0) < rules.min_motion_ratio * (loser.motion or 0.0):
        return QUIET_WINNER
    return None


def pair_weight(delta: float, rules: PairSettings) -> float:
    return min(1.0, delta / rules.margin_ref) if rules.weight_by_margin else 1.0


def _pair(verdict: Judgement, rules: PairSettings, stamp: ScorerStamp) -> PairRecord:
    delta = verdict.score_w - verdict.score_l
    return PairRecord(
        **stamp.model_dump(),
        pair_id=f"{verdict.winner.candidate_id}__vs__{verdict.loser.candidate_id}",
        prompt_id=verdict.winner.prompt_id,
        winner=verdict.winner.candidate_id,
        loser=verdict.loser.candidate_id,
        score_w=verdict.score_w,
        score_l=verdict.score_l,
        margin=delta,
        weight=pair_weight(delta, rules),
        common_checks=sorted(verdict.common),
    )


@dataclass(frozen=True)
class PromptPairs:
    pairs: list[PairRecord]
    dropped: Counter[str]
    rejected: Counter[str]
    eligible: int


def form_pairs(
    scores: Sequence[CandidateScore], rules: PairSettings, rescore: Rescore, stamp: ScorerStamp
) -> PromptPairs:
    """Best against worst eligible, then second best against second worst, ... (rules 2-6)."""
    dropped: Counter[str] = Counter()
    eligible = []
    for score in scores:
        reason = drop_reason(score, rules)
        if reason is None:
            eligible.append(score)
        else:
            dropped[reason] += 1
    rejected: Counter[str] = Counter()
    pairs: list[PairRecord] = []
    ranked = _ranked(eligible)
    if len(ranked) < 2:
        if ranked:
            rejected[SINGLE] += 1
        return PromptPairs(pairs, dropped, rejected, len(eligible))
    for i in range(min(rules.max_pairs_per_prompt, len(ranked) // 2)):
        verdict = judge(ranked[i], ranked[-1 - i], rules, rescore)
        if verdict.rejected is None:
            pairs.append(_pair(verdict, rules, stamp))
        else:
            rejected[verdict.rejected] += 1
    return PromptPairs(pairs, dropped, rejected, len(eligible))


def stamp_of(scores: Sequence[CandidateScore]) -> ScorerStamp:
    stamps = {ScorerStamp(**s.model_dump(include=set(ScorerStamp.model_fields))) for s in scores}
    if len(stamps) != 1:
        raise ValueError(f"scores mix scorers or thresholds: {sorted(map(str, stamps))}")
    return stamps.pop()


def build_pairs(
    prompts: Sequence[PromptRecord],
    scores: Sequence[CandidateScore],
    rules: PairSettings,
    rescore: Rescore,
) -> tuple[list[PairRecord], PairsReport]:
    stamp = stamp_of(scores)
    grouped = by_prompt(scores)
    pairs: list[PairRecord] = []
    dropped: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    eligible = candidates = scored = 0
    for prompt in prompts:
        group = grouped.get(prompt.prompt_id, [])
        result = form_pairs(group, rules, rescore, stamp)
        pairs += result.pairs
        dropped += result.dropped
        rejected += result.rejected
        eligible += result.eligible
        candidates += len(group)
        scored += bool(group)
    report = PairsReport(
        **stamp.model_dump(),
        prompts=len(prompts),
        scored_prompts=scored,
        candidates=candidates,
        eligible=eligible,
        pairs=len(pairs),
        dropped=dict(sorted(dropped.items())),
        rejected=dict(sorted(rejected.items())),
        yield_per_prompt=_per(len(pairs), len(prompts)),
        yield_per_scored_prompt=_per(len(pairs), scored),
    )
    return pairs, report


def _per(count: int, total: int) -> float:
    return count / total if total else 0.0
