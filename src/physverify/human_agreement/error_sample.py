"""A seeded sample of the verifier's false positives and negatives for manual reading.

The ``category`` column stays empty: it is filled by hand with one of ``CATEGORIES``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from physverify.human_agreement.comparison_items import ComparisonItem
from physverify.human_agreement.undecided_policy import policy_view
from physverify.schema.checks import CheckResult

CATEGORIES = (
    "perception_detect",
    "perception_track",
    "perception_depth",
    "perception_scale",
    "physics_not_covered",
    "check_oversensitive",
    "label_noise",
    "other",
)
PROVENANCE_SKIP = ("thresholds",)
Tau = Callable[[str], float | None]


def _providers(results: Sequence[CheckResult], check_id: str | None) -> str:
    chosen = [r for r in results if r.check_id == check_id] or list(results)
    pairs = {
        f"{k}={v}" for r in chosen for k, v in r.provenance.items() if k not in PROVENANCE_SKIP
    }
    return ";".join(sorted(pairs))


def _row(
    item: ComparisonItem, error: str, tau: Tau, results: Sequence[CheckResult]
) -> dict[str, object]:
    check = item.decision.deciding_check
    deciding = next((r for r in item.check_rows if r.check_id == check), None)
    return {
        "clip_id": item.clip_id,
        "version": item.row.version,
        "generator": item.row.generator,
        "caption": item.row.caption,
        "error": error,
        "outcome": item.outcome.value,
        "deciding_check": check or "",
        "magnitude": math.nan
        if deciding is None or deciding.magnitude is None
        else deciding.magnitude,
        "sigma": math.nan if deciding is None else deciding.sigma,
        "tau": (tau(check) if check else None) or math.nan,
        "score": item.decision.score,
        "abstain_reasons": ";".join(item.decision.reasons),
        "providers": _providers(results, check),
        "human_raw": item.row.human_raw,
        "human_violated_rules": item.row.violated_rules,
        "vlm_raw": item.row.reference_raw,
        "vlm_implausible": item.vlm_implausible,
        "category": "",
    }


def sample_errors(
    items: Sequence[ComparisonItem],
    per_class: int,
    seed: int,
    tau: Tau,
    results: Mapping[str, Sequence[CheckResult]],
    policy: str,
) -> list[dict[str, object]]:
    """Up to ``per_class`` false positives and false negatives per version under ``policy``."""
    view = policy_view(items, policy)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for version in sorted({it.row.version for it in items}):
        for error, predicted, human in (("FP", 1, False), ("FN", 0, True)):
            pool = [
                i
                for i, p in zip(view.index, view.verifier, strict=True)
                if items[i].row.version == version
                and p == predicted
                and items[i].human_implausible is human
            ]
            chosen = sorted(rng.permutation(pool)[:per_class].tolist()) if pool else []
            rows.extend(
                _row(items[i], error, tau, results.get(items[i].clip_id, ())) for i in chosen
            )
    return rows
