"""The decision rules of the comparison, declared before the look."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from physverify.baselines.stats import Estimate
from physverify.stats.intervals import Rate

Verdict = Literal["confirmed", "refuted", "no_difference", "not_evaluable"]
CONFIRMED: Verdict = "confirmed"
REFUTED: Verdict = "refuted"
NO_DIFFERENCE: Verdict = "no_difference"
NOT_EVALUABLE: Verdict = "not_evaluable"


def p1(delta: Estimate | None, unread_share: float, max_unread: float) -> Verdict:
    """PA(B1) - PA(judge) on the fresh Kubric set: confirmed when its whole-scene interval is
    above 0; not evaluable when the judge read too few clips or an input is missing."""
    if delta is None or delta.value is None or delta.ci.low is None:
        return NOT_EVALUABLE
    if unread_share > max_unread:
        return NOT_EVALUABLE
    return CONFIRMED if delta.ci.low > 0 else REFUTED


def part2(
    real: Mapping[str, Rate], detection: Estimate | None, max_fpr: float, max_upper: float
) -> Verdict:
    """Per real set the per-clip FPR of B1 > u_B1 within ``max_fpr`` and its upper bound within
    ``max_upper``, and on the fresh Kubric set detection at u_B1 with its lower bound above 0."""
    if not real or detection is None or detection.ci.low is None:
        return NOT_EVALUABLE
    bounds = [(r.value, r.ci.high) for r in real.values()]
    if any(value is None or high is None for value, high in bounds):
        return NOT_EVALUABLE
    quiet = all(
        value is not None and high is not None and value <= max_fpr and high <= max_upper
        for value, high in bounds
    )
    return CONFIRMED if quiet and detection.ci.low > 0 else REFUTED


def part3(low: float | None, high: float | None) -> Verdict:
    """BA(verifier) - BA(VLM) on VideoPhy-2 test: the plan expects the VLM to win (interval
    below 0); an interval that holds 0 is no difference; one above 0 refutes the expectation."""
    if low is None or high is None:
        return NOT_EVALUABLE
    if high < 0:
        return CONFIRMED
    return REFUTED if low > 0 else NO_DIFFERENCE
