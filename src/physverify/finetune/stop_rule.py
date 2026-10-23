"""The pilot's stop rule (ADR 0012, item 1.6): when it fires, H3 is not evaluable with this
verifier, and training or the eval arm is refused unless a deviation is declared.

It fires on too few pairs per scored prompt, too many abstaining candidates, or a hand review
(``finetune pilot-review``) of other pairs or with too few better winners. With fewer pairs than
``inspected`` all of them are inspected, and ``ceil(min_better_fraction * n)`` must be better.
The inspection itself stays manual: the code only requires that it is recorded, on these pairs.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import Field

from physverify.core.digests import text_sha256
from physverify.core.models import FrozenModel
from physverify.finetune.candidate_scoring import same_stamp
from physverify.finetune.layout import (
    PAIRS_FILE,
    PILOT_REVIEW_FILE,
    REPORT_FILE,
    STOP_RULE_FILE,
    CampaignStore,
)
from physverify.finetune.pilot import fit_and_val
from physverify.finetune.records import (
    BASE_TAG,
    CandidateScore,
    PairRecord,
    PairsReport,
    read_record,
    read_records,
    write_record,
)
from physverify.finetune.scorers._base import AggregateScorer
from physverify.finetune.settings import StopRuleSettings

ROUNDING: Final = 9  # the configured edges (0.3 pairs, 0.5 abstention, 0.6 x n) pass exactly


class PilotReview(FrozenModel):
    """``better`` of the ``inspected`` pair ids show a physically better winner."""

    pairs_sha256: str
    inspected: list[str]
    better: int = Field(ge=0)
    note: str | None = None
    created_utc: str


class StopVerdict(FrozenModel):
    passed: bool
    enforced: bool
    reasons: list[str]
    measured: dict[str, float | int | None]
    pairs_sha256: str
    thresholds: str | None
    families: dict[str, list[str]]
    deviation: str | None = None


class StopRuleState(FrozenModel):
    """What H3 freezes of the verdict: recomputable from disk, so a later change is caught."""

    passed: bool
    enforced: bool
    reasons: list[str]
    pairs_sha256: str


def frozen_state(verdict: StopVerdict) -> StopRuleState:
    return StopRuleState(
        passed=verdict.passed,
        enforced=verdict.enforced,
        reasons=verdict.reasons,
        pairs_sha256=verdict.pairs_sha256,
    )


def fired_reason(state: StopRuleState, deviation: str | None) -> str | None:
    """Why H3 is not evaluable with this verifier; None when the rule passed."""
    if state.passed:
        return None
    declared = "none" if deviation is None else repr(deviation)
    return f"pilot stop rule fired ({'; '.join(state.reasons)}); deviation declared: {declared}"


def pairs_sha256(pairs: Sequence[PairRecord]) -> str:
    """The hash of the pairs as stored (one JSON line each)."""
    text = "".join(p.model_dump_json() + "\n" for p in pairs)
    return text_sha256(text)


def inspected_count(n_pairs: int, rules: StopRuleSettings) -> int:
    return min(rules.inspected, n_pairs)


def required_better(n_inspected: int, rules: StopRuleSettings) -> int:
    return math.ceil(round(rules.min_better_fraction * n_inspected, ROUNDING))


def review_sample(pairs: Sequence[PairRecord], rules: StopRuleSettings) -> list[PairRecord]:
    """The pairs to inspect: a seeded draw of ``inspected`` (all of them when fewer)."""
    count = inspected_count(len(pairs), rules)
    chosen = random.Random(rules.seed).sample(range(len(pairs)), count)
    return [pairs[i] for i in sorted(chosen)]


def candidate_abstention(scores: Sequence[CandidateScore]) -> float | None:
    """Fraction of scored candidates whose score is None; None without candidates."""
    if not scores:
        return None
    return sum(score.score is None for score in scores) / len(scores)


def _review_reasons(
    review: PilotReview | None, pairs: Sequence[PairRecord], sha256: str, rules: StopRuleSettings
) -> list[str]:
    if review is None:
        return ["no pilot review recorded: run 'finetune pilot-review'"]
    if review.pairs_sha256 != sha256:
        return ["the pilot review was made on other pairs: review the current ones"]
    expected = {p.pair_id for p in review_sample(pairs, rules)}
    if set(review.inspected) != expected:
        return [f"the pilot review inspected {len(review.inspected)} pairs, not the drawn sample"]
    needed = required_better(len(expected), rules)
    if review.better < needed:
        return [
            f"{review.better} of {len(expected)} inspected pairs show a better winner (< {needed})"
        ]
    return []


def _stamp_reasons(
    report: PairsReport | None, pairs: Sequence[PairRecord], stamp: tuple[str, str]
) -> list[str]:
    """The pairs and their report must come from the scorer as it is configured now."""
    stamps = {(r.scorer_version, r.thresholds) for r in [*pairs, *([report] if report else [])]}
    if stamps - {stamp}:
        return ["the pairs were built under another scorer stamp: run 'finetune pairs' again"]
    return []


def evaluate_stop(
    report: PairsReport | None,
    pairs: Sequence[PairRecord],
    abstention: float | None,
    review: PilotReview | None,
    rules: StopRuleSettings,
    families: Mapping[str, list[str]],
    stamp: tuple[str, str],
) -> StopVerdict:
    """Every reason the rule fires; ``passed`` without any. ``stamp`` is the active scorer's
    (version, thresholds); ``abstention`` is over its scores of the scored fit prompts."""
    sha256 = pairs_sha256(pairs)
    reasons = _stamp_reasons(report, pairs, stamp)
    measured = None if report is None else report.measured_yield
    if measured is None or round(measured, ROUNDING) < rules.min_pairs_per_prompt:
        reasons.append(f"pairs per scored prompt {measured} < {rules.min_pairs_per_prompt}")
    if abstention is None or round(abstention, ROUNDING) > rules.max_candidate_abstention:
        reasons.append(f"candidate abstention {abstention} > {rules.max_candidate_abstention}")
    reasons += _review_reasons(review, pairs, sha256, rules)
    return StopVerdict(
        passed=not reasons,
        enforced=rules.enforce,
        reasons=reasons,
        measured={
            "pairs_per_scored_prompt": measured,
            "candidate_abstention": abstention,
            "pairs": len(pairs),
            "inspected": None if review is None else len(review.inspected),
            "better": None if review is None else review.better,
        },
        pairs_sha256=sha256,
        thresholds=None if report is None else report.thresholds,
        families=dict(families),
    )


def pairs_on_disk(store: CampaignStore, scorer: str) -> list[PairRecord]:
    path = store.pairs_dir(scorer) / PAIRS_FILE
    return read_records(path, PairRecord) if path.is_file() else []


def stored_review(store: CampaignStore, scorer: str) -> PilotReview | None:
    path = store.pairs_dir(scorer) / PILOT_REVIEW_FILE
    return read_record(path, PilotReview) if path.is_file() else None


def stored_report(store: CampaignStore, scorer: str) -> PairsReport | None:
    path = store.pairs_dir(scorer) / REPORT_FILE
    return read_record(path, PairsReport) if path.is_file() else None


def base_abstention(store: CampaignStore, scorer: AggregateScorer) -> float | None:
    """Over the active scorer's base scores of the fit prompts (the pilot's)."""
    train = (
        [p for p in store.prompts() if p.split == "train"] if store.prompts_path.is_file() else []
    )
    fit_ids = {p.prompt_id for p in fit_and_val(train)[0]}
    scores = store.scores(BASE_TAG, scorer.name)
    return candidate_abstention(
        [s for s in scores if s.prompt_id in fit_ids and same_stamp(s, scorer)]
    )


def write_verdict(store: CampaignStore, scorer: str, verdict: StopVerdict) -> None:
    write_record(store.pairs_dir(scorer) / STOP_RULE_FILE, verdict)


def verdict_lines(verdict: StopVerdict) -> list[str]:
    state = "passed" if verdict.passed else "FIRED"
    mode = "enforced" if verdict.enforced else "reported only"
    lines = [f"pilot stop rule: {state} ({mode}); measured {verdict.measured}"]
    lines += [f"  reason: {reason}" for reason in verdict.reasons]
    lines += [f"  family {name}: targets {kept}" for name, kept in verdict.families.items()]
    if verdict.deviation is not None:
        lines.append(f"  declared deviation: {verdict.deviation}")
    return lines
