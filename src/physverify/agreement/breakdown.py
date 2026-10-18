"""Where the verifier's agreement comes from (ADR 0011 D9): per check, per generator."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from physverify.agreement.abstention import AS_PLAUSIBLE, labels_of, policy_view
from physverify.agreement.items import ComparisonItem
from physverify.agreement.stats import accuracy, balanced_accuracy
from physverify.schema.checks import Verdict
from physverify.verification.runner import RowStatus

# The pooled group H2 is decided on; each version alone is descriptive (too few pairs).
AGGREGATE = "all"

Redecide = Callable[[frozenset[str]], Sequence[ComparisonItem]]


def _fails(item: ComparisonItem, check_id: str) -> bool:
    return any(r.check_id == check_id and r.verdict is Verdict.FAIL for r in item.check_rows)


def _fail_rate(items: Sequence[ComparisonItem], check_id: str, human: bool) -> float:
    group = [it for it in items if it.human_implausible is human]
    return sum(_fails(it, check_id) for it in group) / len(group) if group else float("nan")


def primary_accuracy(items: Sequence[ComparisonItem]) -> float:
    view = policy_view(items, AS_PLAUSIBLE)
    return accuracy(labels_of(items, view.index, lambda it: it.human_implausible), view.verifier)


def per_check(items: Sequence[ComparisonItem], redecide: Redecide) -> list[dict[str, object]]:
    """n scored, FAIL rate by human label, the check alone (FAIL = implausible), and the
    accuracy of the rule without it."""
    check_ids = sorted({r.check_id for it in items for r in it.check_rows})
    y = labels_of(items, range(len(items)), lambda it: it.human_implausible)
    rows: list[dict[str, object]] = []
    for check_id in check_ids:
        alone = np.asarray([int(_fails(it, check_id)) for it in items], dtype=np.int_)
        rows.append(
            {
                "check_id": check_id,
                "n_scored": sum(
                    r.check_id == check_id and r.status is RowStatus.SCORED
                    for it in items
                    for r in it.check_rows
                ),
                "fail_rate_human_implausible": _fail_rate(items, check_id, True),
                "fail_rate_human_plausible": _fail_rate(items, check_id, False),
                "accuracy_alone": accuracy(y, alone),
                "balanced_accuracy_alone": balanced_accuracy(y, alone),
                "accuracy_without": primary_accuracy(redecide(frozenset({check_id}))),
            }
        )
    return rows


def per_group(
    items: Sequence[ComparisonItem], key: Callable[[ComparisonItem], str]
) -> list[dict[str, object]]:
    """Primary-policy accuracy of the verifier and the VLM within each group."""
    groups: dict[str, list[ComparisonItem]] = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    rows: list[dict[str, object]] = []
    for name, group in sorted(groups.items()):
        view = policy_view(group, AS_PLAUSIBLE)
        y = labels_of(group, view.index, lambda it: it.human_implausible)
        vlm = labels_of(group, view.index, lambda it: it.vlm_implausible)
        rows.append(
            {
                "group": name,
                "n": len(view.index),
                "human_implausible_rate": float(np.mean(y)) if y.size else float("nan"),
                "verifier_accuracy": accuracy(y, view.verifier),
                "vlm_accuracy": accuracy(y, vlm),
                "verifier_balanced_accuracy": balanced_accuracy(y, view.verifier),
                "vlm_balanced_accuracy": balanced_accuracy(y, vlm),
            }
        )
    return rows


def by_version(items: Sequence[ComparisonItem]) -> Mapping[str, list[ComparisonItem]]:
    groups: dict[str, list[ComparisonItem]] = {AGGREGATE: list(items)}
    for item in items:
        groups.setdefault(item.row.version, []).append(item)
    return groups
