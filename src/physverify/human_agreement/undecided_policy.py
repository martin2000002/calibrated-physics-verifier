"""What an undecided verifier clip counts as. Each policy returns the indices of the
items it scores and, per item, the verifier and the VLM predictions (1 = implausible)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from physverify.human_agreement.clip_decision import DECIDED, ClipDecision
from physverify.human_agreement.comparison_items import ComparisonItem
from physverify.human_agreement.stats import Labels, accuracy, as_labels

AS_PLAUSIBLE = "as_plausible"
DECIDED_ONLY = "decided_only"
AS_IMPLAUSIBLE = "as_implausible"
COVERAGE_MATCHED_VLM = "coverage_matched_vlm"
CURVE_STEPS = 10


@dataclass(frozen=True)
class PolicyView:
    """``vlm_index`` differs from ``index`` only for coverage_matched_vlm (unpaired)."""

    name: str
    index: tuple[int, ...]
    verifier: Labels
    vlm_index: tuple[int, ...]
    paired: bool


def _verifier(items: Sequence[ComparisonItem], index: Sequence[int], undecided: int) -> Labels:
    def label(item: ComparisonItem) -> int:
        if item.outcome in DECIDED:
            return int(item.outcome is ClipDecision.IMPLAUSIBLE)
        return undecided

    return np.asarray([label(items[i]) for i in index], dtype=np.int_)


def with_vlm(items: Sequence[ComparisonItem]) -> tuple[int, ...]:
    return tuple(i for i, item in enumerate(items) if item.vlm_implausible is not None)


def decided(items: Sequence[ComparisonItem], index: Sequence[int]) -> tuple[int, ...]:
    return tuple(i for i in index if items[i].outcome in DECIDED)


def vlm_confidence(item: ComparisonItem) -> float:
    return abs(item.row.reference_raw - item.row.reference_cut)


def most_confident(
    items: Sequence[ComparisonItem], index: Sequence[int], k: int
) -> tuple[int, ...]:
    """The ``k`` items where the VLM is furthest from its cut (ties by position)."""
    ranked = sorted(index, key=lambda i: (-vlm_confidence(items[i]), i))
    return tuple(sorted(ranked[:k]))


def policy_view(items: Sequence[ComparisonItem], name: str) -> PolicyView:
    """Items with a VLM prediction, so every policy compares the same population."""
    paired = with_vlm(items)
    if name == AS_PLAUSIBLE:
        return PolicyView(name, paired, _verifier(items, paired, 0), paired, True)
    if name == AS_IMPLAUSIBLE:
        return PolicyView(name, paired, _verifier(items, paired, 1), paired, True)
    chosen = decided(items, paired)
    if name == DECIDED_ONLY:
        return PolicyView(name, chosen, _verifier(items, chosen, 0), chosen, True)
    if name == COVERAGE_MATCHED_VLM:
        matched = most_confident(items, paired, len(chosen))
        return PolicyView(name, chosen, _verifier(items, chosen, 0), matched, False)
    raise ValueError(f"unknown abstention policy {name!r}")


POLICIES = (AS_PLAUSIBLE, DECIDED_ONLY, AS_IMPLAUSIBLE, COVERAGE_MATCHED_VLM)


def labels_of(
    items: Sequence[ComparisonItem],
    index: Sequence[int],
    get: Callable[[ComparisonItem], bool | None],
) -> Labels:
    return as_labels([bool(get(items[i])) for i in index])


def risk_coverage(items: Sequence[ComparisonItem]) -> list[dict[str, float]]:
    """VLM accuracy on its most confident fraction of items, in steps of 1/CURVE_STEPS."""
    paired = with_vlm(items)
    curve = []
    for step in range(1, CURVE_STEPS + 1):
        kept = most_confident(items, paired, round(len(paired) * step / CURVE_STEPS))
        y = labels_of(items, kept, lambda it: it.human_implausible)
        p = labels_of(items, kept, lambda it: it.vlm_implausible)
        curve.append(
            {"coverage": step / CURVE_STEPS, "n": float(len(kept)), "accuracy": accuracy(y, p)}
        )
    return curve


def abstention_counts(items: Sequence[ComparisonItem]) -> dict[str, dict[str, int]]:
    """Outcomes, and abstention reasons per clip (INTERVAL_CROSSES_THRESHOLD is one of them)."""
    outcomes = Counter(item.outcome.value for item in items)
    reasons = Counter(
        reason for item in items if item.outcome not in DECIDED for reason in item.decision.reasons
    )
    return {"outcomes": dict(sorted(outcomes.items())), "reasons": dict(sorted(reasons.items()))}
