"""The clip verdict of one check, shared by ``fit`` and ``apply``.

A check yields one result per subject. The clip verdict is the run table's (``row_verdict``) and
its magnitude that of the ``decisive`` subject, as in the table. Fit-time rates, weights and
Platt fits use the same outcome that ``score_clip`` applies, so what the artifact reports is what
deployment does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from physverify.schema.checks import AbstainReason, CheckResult, Verdict
from physverify.verification.decisive import decisive
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.runner import ThresholdEntry, row_verdict

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
    """A check's verdict on a clip and its ``decisive`` subject. ``magnitude`` is m - bias and
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
    """The verdict of the set (``row_verdict``) with the magnitude, sigma and ``result`` of the
    ``decisive`` subject: an abstained clip reports its largest non-abstained subject when it has
    one, and is ``judged`` only when decided. Its reason is that of the largest abstained subject,
    where the run table keeps the most common one. ``NO_EVENT`` for no result. ``tau`` is explicit
    because the cross-validated FPR judges each fold with its own tau."""
    outcomes = [subject_outcome(r, rule, tau) for r in results]
    chosen = decisive(outcomes)
    verdict = row_verdict(outcomes)
    if chosen is None or verdict is None:
        return NO_EVENT
    if verdict is chosen.verdict:
        return chosen
    judged = chosen.judged and verdict in DECIDED
    return _with(chosen, verdict, _abstain_reason(outcomes), judged=judged)


def _abstain_reason(outcomes: Sequence[Outcome]) -> AbstainReason | None:
    abstained = decisive([o for o in outcomes if o.verdict is Verdict.ABSTAIN])
    return None if abstained is None else abstained.abstain_reason
