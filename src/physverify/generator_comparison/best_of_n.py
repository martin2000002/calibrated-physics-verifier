"""Best-of-N under the frozen rule: the highest-scoring eligible base candidate.

Eligible means ``pairs.drop_reason`` is None, the same filter the preference pairs use. A tie goes
to the smaller k. With no eligible candidate the best scored one is taken (``ineligible``); with
no score at all, k = 0 (``all_abstained``). Both fallbacks are counted in the report.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from physverify.finetune.pairs import drop_reason
from physverify.finetune.records import CandidateScore
from physverify.finetune.settings import PairSettings

Fallback = Literal["ineligible", "all_abstained"]
Key = Callable[[CandidateScore], float | None]


def stored_score(score: CandidateScore) -> float | None:
    return score.score


@dataclass(frozen=True)
class Pick:
    k: int
    score: CandidateScore
    value: float | None
    n_eligible: int
    fallback: Fallback | None


def _best(values: Mapping[int, float]) -> int:
    return max(values, key=lambda k: (values[k], -k))


def pick_best(
    candidates: Mapping[int, CandidateScore], rules: PairSettings, key: Key = stored_score
) -> Pick | None:
    """None only when the prompt has no candidate at all."""
    if not candidates:
        return None
    keyed = {k: key(s) for k, s in candidates.items()}
    scored = {k: v for k, v in keyed.items() if v is not None}
    eligible = {k: v for k, v in scored.items() if drop_reason(candidates[k], rules) is None}
    if eligible:
        k = _best(eligible)
        return Pick(k, candidates[k], eligible[k], len(eligible), None)
    if scored:
        k = _best(scored)
        return Pick(k, candidates[k], scored[k], 0, "ineligible")
    k = min(candidates)
    return Pick(k, candidates[k], None, 0, "all_abstained")
