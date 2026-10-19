"""The clip verdict of one check, shared by ``fit`` and ``apply`` (ADR 0009 D5, D11).

A check yields one result per subject. The clip outcome is its worst FAIL, else its largest PASS
when every subject passes, else an abstention. Fit-time rates, weights and Platt fits use the same
outcome that ``score_clip`` applies, so what the artifact reports is what deployment does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from physverify.schema.checks import AbstainReason, CheckResult, Verdict
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.runner import ThresholdEntry

DECIDED = (Verdict.PASS, Verdict.FAIL)


@dataclass(frozen=True)
class Rule:
    """The verdict rule of ``ThresholdPolicy`` with the measured bias and sigma factor."""

    policy: ThresholdPolicy
    bias: float
    factor: float | None

    def entry(self, tau: float) -> ThresholdEntry:
        return ThresholdEntry(tau=tau, bias=self.bias, sigma_factor=self.factor)


@dataclass(frozen=True)
class Outcome:
    """A check's verdict on a clip and the subject that decides it. ``magnitude`` is m - bias and
    ``sigma`` factor * sigma (NaN: not measured); ``judged`` is True when the policy compared the
    subject with tau; ``result`` is None when the check had no event on the clip."""

    verdict: Verdict
    abstain_reason: AbstainReason | None
    magnitude: float | None
    sigma: float
    judged: bool
    result: CheckResult | None


NO_EVENT = Outcome(Verdict.UNSCORED, None, None, math.nan, False, None)


def subject_outcome(result: CheckResult, rule: Rule, tau: float | None) -> Outcome:
    """One subject: ABSTAIN when it has no magnitude, UNSCORED without tau, else the policy."""
    factor = math.nan if rule.factor is None else rule.factor
    magnitude = None if result.magnitude is None else result.magnitude - rule.bias
    base = Outcome(Verdict.UNSCORED, None, magnitude, factor * result.sigma, False, result)
    if result.verdict is Verdict.ABSTAIN or result.magnitude is None:
        return _with(base, Verdict.ABSTAIN, result.abstain_reason, judged=False)
    if tau is None:
        return base
    verdict, reason = rule.policy.judge(result.magnitude, result.sigma, rule.entry(tau))
    return _with(base, verdict, reason, judged=True)


def _with(
    base: Outcome, verdict: Verdict, reason: AbstainReason | None, *, judged: bool
) -> Outcome:
    return Outcome(verdict, reason, base.magnitude, base.sigma, judged, base.result)


def clip_verdict(results: Sequence[CheckResult], rule: Rule, tau: float | None) -> Outcome:
    """The worst FAIL, else the largest PASS when every subject passes, else the undecided subject
    of largest magnitude (``NO_EVENT`` for no result). ``tau`` is explicit because the
    cross-validated FPR judges each fold with its own tau."""
    outcomes = [subject_outcome(r, rule, tau) for r in results]
    if not outcomes:
        return NO_EVENT
    failed = [o for o in outcomes if o.verdict is Verdict.FAIL]
    if failed:
        return max(failed, key=_size)
    if all(o.verdict is Verdict.PASS for o in outcomes):
        return max(outcomes, key=_size)
    return max((o for o in outcomes if o.verdict not in DECIDED), key=_size)


def _size(outcome: Outcome) -> float:
    return -math.inf if outcome.magnitude is None else outcome.magnitude
