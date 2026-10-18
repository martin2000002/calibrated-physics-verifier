"""Calibrated verdicts, violation probabilities and the clip score (ADR 0009 D11, Juicio 6-8).

What A9, A10 and A11 import. The verdict rule is ``ThresholdPolicy.judge`` (no copy of k sigma);
p_c is an estimate conditional on the calib design and S inherits that label.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from scipy.special import expit

from physverify.calibration.loader import KSigmaMismatchError, Stratum
from physverify.calibration.schema import CheckCalibration
from physverify.calibration.thresholds import DECIDED
from physverify.schema.checks import AbstainReason, CheckResult, Verdict
from physverify.verification.policies.threshold import ThresholdParams, ThresholdPolicy

NO_DECIDED = "no_decided_check"
NO_INFORMATIVE = "no_informative_check"
P_MAX = 1.0 - 1e-12


@dataclass(frozen=True)
class CalibratedResult:
    """``magnitude`` is m - bias and ``sigma`` factor * sigma (NaN: not measured)."""

    check_id: str
    magnitude: float | None
    sigma: float
    tau: float | None
    p_violation: float | None
    verdict: Verdict
    abstain_reason: AbstainReason | None


def stratum_k_sigma(stratum: Stratum, k_sigma: float | None = None) -> float:
    """The k of the fit. tau, p_c and the weights were fitted with it, so another k is an error."""
    fitted = stratum.model.k_sigma
    if k_sigma is not None and not math.isclose(k_sigma, fitted, rel_tol=1e-12, abs_tol=1e-12):
        raise KSigmaMismatchError(
            f"k_sigma {k_sigma} differs from the {fitted} the stratum {stratum.source} was "
            "calibrated with; refit or omit it"
        )
    return fitted


def policy_for(stratum: Stratum, k_sigma: float | None = None) -> ThresholdPolicy:
    params = ThresholdParams(
        k_sigma=stratum_k_sigma(stratum, k_sigma), on_unknown_sigma=stratum.model.on_unknown_sigma
    )
    return ThresholdPolicy(params)


def violation_probability(check: CheckCalibration, magnitude: float) -> float | None:
    """p = expit(a + b u), u = (m - bias - tau) / s: an estimate. None when not fitted."""
    prob = check.probability
    if prob.a is None or prob.b is None or check.tau is None:
        return None
    u = (magnitude - check.sigma.bias - check.tau) / prob.scale
    return float(expit(prob.a + prob.b * u))


def calibrated_result(
    result: CheckResult, stratum: Stratum, k_sigma: float | None = None
) -> CalibratedResult:
    return _calibrated(result, stratum, policy_for(stratum, k_sigma))


def _calibrated(result: CheckResult, stratum: Stratum, policy: ThresholdPolicy) -> CalibratedResult:
    bias, factor = stratum.sigma_correction(result.check_id)
    entry = stratum.entry(result.check_id)
    magnitude = None if result.magnitude is None else result.magnitude - bias
    base = CalibratedResult(
        check_id=result.check_id,
        magnitude=magnitude,
        sigma=factor * result.sigma,
        tau=None if entry is None else entry.tau,
        p_violation=None,
        verdict=Verdict.UNSCORED,
        abstain_reason=None,
    )
    if result.verdict is Verdict.ABSTAIN or result.magnitude is None:
        return replace(base, verdict=Verdict.ABSTAIN, abstain_reason=result.abstain_reason)
    if entry is None:
        return base
    verdict, reason = policy.judge(result.magnitude, result.sigma, entry)
    check = stratum.check(result.check_id)
    p = None if check is None else violation_probability(check, result.magnitude)
    return replace(base, verdict=verdict, abstain_reason=reason, p_violation=p)


@dataclass(frozen=True)
class ClipScore:
    """S = prod_{c in D} (1 - p_c)^{w_c} (1 = no evidence of violation); ``sigma`` by the delta
    method (0 on the oracle). ``coverage`` = sum_D w / sum_A w over checks with an event."""

    clip_id: str
    score: float | None
    sigma: float | None
    coverage: float
    abstained: bool
    reason: str | None
    n_failed: int
    decided_checks: tuple[str, ...]
    abstained_checks: tuple[str, ...]
    per_check: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class _Decided:
    check_id: str
    weight: float
    p: float
    dp_dm: float
    sigma: float
    failed: bool


def check_outcome(
    results: Sequence[CheckResult], stratum: Stratum, policy: ThresholdPolicy
) -> CalibratedResult | None:
    """The check's clip outcome: its worst FAIL, else its largest PASS when every scored subject
    passes, else an abstention (None when it has no event)."""
    calibrated = [_calibrated(r, stratum, policy) for r in results]
    if not calibrated:
        return None
    failed = [c for c in calibrated if c.verdict is Verdict.FAIL]
    if failed:
        return max(failed, key=lambda c: c.magnitude or 0.0)
    if all(c.verdict is Verdict.PASS for c in calibrated):
        return max(calibrated, key=lambda c: c.magnitude or 0.0)
    return next(c for c in calibrated if c.verdict not in DECIDED)


def _derivative(check: CheckCalibration, p: float) -> float:
    """dp/dm = p (1 - p) b / s."""
    b = check.probability.b or 0.0
    return p * (1.0 - p) * b / check.probability.scale


def score_clip(
    results: Sequence[CheckResult],
    stratum: Stratum,
    k_sigma: float | None = None,
    only: Collection[str] | None = None,
) -> ClipScore:
    """Score of one clip's results; ``only`` restricts the checks (A11 compares two videos on the
    checks decided in both). ``k_sigma`` defaults to the stratum's; a different one raises
    KSigmaMismatchError."""
    policy = policy_for(stratum, k_sigma)
    by_check: dict[str, list[CheckResult]] = {}
    for result in results:
        if stratum.tau(result.check_id) is not None and (only is None or result.check_id in only):
            by_check.setdefault(result.check_id, []).append(result)
    clip_id = results[0].clip_id if results else ""
    decided, abstained = _split(by_check, stratum, policy)
    return _score(clip_id, decided, abstained, stratum)


def _split(
    by_check: Mapping[str, Sequence[CheckResult]], stratum: Stratum, policy: ThresholdPolicy
) -> tuple[list[_Decided], list[str]]:
    decided: list[_Decided] = []
    abstained: list[str] = []
    for check_id in sorted(by_check):
        outcome = check_outcome(by_check[check_id], stratum, policy)
        check = stratum.check(check_id)
        if outcome is None or check is None:
            continue
        if outcome.verdict not in DECIDED or outcome.p_violation is None:
            abstained.append(check_id)
            continue
        p = min(outcome.p_violation, P_MAX)
        decided.append(
            _Decided(
                check_id=check_id,
                weight=stratum.weight(check_id),
                p=p,
                dp_dm=_derivative(check, p),
                sigma=outcome.sigma,
                failed=outcome.verdict is Verdict.FAIL,
            )
        )
    return decided, abstained


def _score(
    clip_id: str, decided: Sequence[_Decided], abstained: Sequence[str], stratum: Stratum
) -> ClipScore:
    total = sum(d.weight for d in decided) + sum(stratum.weight(c) for c in abstained)
    base = ClipScore(
        clip_id=clip_id,
        score=None,
        sigma=None,
        coverage=sum(d.weight for d in decided) / total if total > 0.0 else 0.0,
        abstained=True,
        reason=None,
        n_failed=sum(d.failed for d in decided),
        decided_checks=tuple(d.check_id for d in decided),
        abstained_checks=tuple(abstained),
        per_check=MappingProxyType({d.check_id: d.p for d in decided}),
    )
    if stratum.max_weight() <= 0.0:
        return replace(base, reason=NO_INFORMATIVE)
    if not decided:
        return replace(base, reason=NO_DECIDED)
    score = math.prod((1.0 - d.p) ** d.weight for d in decided)
    return replace(base, score=score, sigma=score_sigma(score, decided), abstained=False)


def score_sigma(score: float, decided: Sequence[_Decided]) -> float | None:
    """sigma_S^2 = sum_c (S w_c p'_c sigma_c / (1 - p_c))^2; None when a sigma is unknown."""
    terms = [score * d.weight * d.dp_dm * d.sigma / (1.0 - d.p) for d in decided if d.weight > 0]
    if any(not math.isfinite(t) for t in terms):
        return None
    return math.sqrt(sum(t * t for t in terms))
