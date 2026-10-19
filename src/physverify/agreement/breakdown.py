"""Where the verifier's agreement comes from (ADR 0011 D9): per check, per generator."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence

import numpy as np

from physverify.agreement.abstention import labels_of, policy_view
from physverify.agreement.decision import ClipDecision
from physverify.agreement.items import ComparisonItem
from physverify.agreement.stats import accuracy, balanced_accuracy
from physverify.verification.runner import RowStatus

# The pooled group H2 is decided on; each version alone is descriptive (too few pairs).
AGGREGATE = "all"

Redecide = Callable[[frozenset[str]], Sequence[ComparisonItem]]


def _human(item: ComparisonItem) -> bool:
    return item.human_implausible


def _implausible_rate(items: Sequence[ComparisonItem], human: bool) -> float:
    group = [it for it in items if it.human_implausible is human]
    hits = sum(it.outcome is ClipDecision.IMPLAUSIBLE for it in group)
    return hits / len(group) if group else float("nan")


def policy_accuracy(items: Sequence[ComparisonItem], policy: str) -> float:
    view = policy_view(items, policy)
    return accuracy(labels_of(items, view.index, _human), view.verifier)


def _scored(items: Sequence[ComparisonItem], check_id: str) -> int:
    return sum(
        r.check_id == check_id and r.status is RowStatus.SCORED
        for it in items
        for r in it.check_rows
    )


def per_check(
    items: Sequence[ComparisonItem],
    redecide: Redecide,
    check_ids: Collection[str],
    policy: str,
) -> list[dict[str, object]]:
    """Per check: n scored, and the compared rule decided on that check alone (every other check
    excluded, so a calibrated rule uses its own tau and the frozen cut, never the verify run's
    provisional verdicts): its implausible rate by human label and its accuracy under
    ``policy``; then the accuracy of the rule without the check. ``check_ids`` is every check the
    rule sees (table rows and results)."""
    every = frozenset(check_ids)
    rows: list[dict[str, object]] = []
    for check_id in sorted(every):
        alone = redecide(every - {check_id})
        view = policy_view(alone, policy)
        y = labels_of(alone, view.index, _human)
        rows.append(
            {
                "check_id": check_id,
                "n_scored": _scored(items, check_id),
                "implausible_rate_alone_human_implausible": _implausible_rate(alone, True),
                "implausible_rate_alone_human_plausible": _implausible_rate(alone, False),
                "accuracy_alone": accuracy(y, view.verifier),
                "balanced_accuracy_alone": balanced_accuracy(y, view.verifier),
                "accuracy_without": policy_accuracy(redecide(frozenset({check_id})), policy),
            }
        )
    return rows


def per_group(
    items: Sequence[ComparisonItem], key: Callable[[ComparisonItem], str], policy: str
) -> list[dict[str, object]]:
    """Accuracy of the verifier and the VLM within each group, under ``policy``."""
    groups: dict[str, list[ComparisonItem]] = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    rows: list[dict[str, object]] = []
    for name, group in sorted(groups.items()):
        view = policy_view(group, policy)
        y = labels_of(group, view.index, _human)
        y_vlm = labels_of(group, view.vlm_index, _human)
        vlm = labels_of(group, view.vlm_index, lambda it: it.vlm_implausible)
        rows.append(
            {
                "group": name,
                "n": len(view.index),
                "human_implausible_rate": float(np.mean(y)) if y.size else float("nan"),
                "verifier_accuracy": accuracy(y, view.verifier),
                "vlm_accuracy": accuracy(y_vlm, vlm),
                "verifier_balanced_accuracy": balanced_accuracy(y, view.verifier),
                "vlm_balanced_accuracy": balanced_accuracy(y_vlm, vlm),
            }
        )
    return rows


def by_version(items: Sequence[ComparisonItem]) -> Mapping[str, list[ComparisonItem]]:
    groups: dict[str, list[ComparisonItem]] = {AGGREGATE: list(items)}
    for item in items:
        groups.setdefault(item.row.version, []).append(item)
    return groups
