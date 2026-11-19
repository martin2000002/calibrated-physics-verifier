"""Graded score: every check that orders severity votes with its calibrated violation probability,
whether or not its interval clears tau.

``S = prod_{c in G} (1 - p_c)^{w_c}`` over the graded checks G. A check grades a clip from its
largest subject with a finite magnitude whose verdict is decided, unscored or an abstention only
because the interval crosses tau; a structural abstention (no input, occluded, ...) does not grade.
``p_c`` is the stratum's Platt estimate at that magnitude. The weight is the lower bound of the
within-scene Kendall tau_b of level against magnitude, clamped at 0 and normalised by the largest:
pairs rank candidates of one prompt, so a check votes by how well it orders severities, not by how
well it separates clean from violated clips (the calibrated score's weight). A check without a
measured sigma factor does not grade. S and its delta-method sigma come from the calibration
chain's ``weighted_score``, so ``pairs.sigma_k`` applies unchanged; ``decided`` lists G, so pairs
compare on the checks graded in both candidates. ``n_failed`` counts the graded checks whose
point magnitude exceeds tau (no interval), not FAIL verdicts. The stratum is opened through
``_calibrated_gate`` (same artifact, stamp and time-base checks as the calibrated score)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Final

from physverify.finetune.scorers._base import AggregateScore
from physverify.finetune.scorers._calibrated_gate import CalibratedGate
from physverify.schema.checks import AbstainReason, CheckResult, Verdict

if TYPE_CHECKING:
    from physverify.calibration.api import CalibrationApi
    from physverify.calibration.apply import Vote
    from physverify.calibration.artifact import CheckCalibration
    from physverify.calibration.artifact_io import Stratum

NO_GRADED: Final = "no_graded_check"
GRADING_REASONS: Final = frozenset({None, AbstainReason.INTERVAL_CROSSES_THRESHOLD})

WeightOf = Callable[["CheckCalibration"], float]


def monotonicity_weight(check: CheckCalibration) -> float:
    low = check.monotonicity.ci.low
    return 0.0 if low is None or not math.isfinite(low) else max(0.0, low)


def grades(result: CheckResult) -> bool:
    if result.magnitude is None or not math.isfinite(result.magnitude):
        return False
    if not math.isfinite(result.sigma):
        return False
    if result.verdict is Verdict.ABSTAIN:
        return result.abstain_reason in GRADING_REASONS
    return True


def graded_vote(
    api: CalibrationApi[Stratum],
    check_id: str,
    results: Sequence[CheckResult],
    check: CheckCalibration,
    weight: float,
) -> Vote | None:
    usable = [r for r in results if grades(r)]
    factor = check.sigma.factor
    if not usable or weight <= 0.0 or factor is None or not math.isfinite(factor):
        return None
    top = max(usable, key=lambda r: r.magnitude or 0.0)
    magnitude = top.magnitude or 0.0
    p = api.violation_probability(check, magnitude)
    if p is None:
        return None
    exceeds = check.tau is not None and magnitude - check.sigma.bias > check.tau
    return api.vote(check_id, check, p, weight, factor * top.sigma, exceeds)


def stratum_weights(stratum: Stratum, weight_of: WeightOf) -> dict[str, float]:
    """Normalised by the largest; only checks with status ok and a tau."""
    raw = {
        check_id: weight_of(check)
        for check_id, check in stratum.model.checks.items()
        if stratum.tau(check_id) is not None
    }
    top = max(raw.values(), default=0.0)
    return {c: (w / top if top > 0.0 else 0.0) for c, w in raw.items()}


def graded_score(
    api: CalibrationApi[Stratum],
    results: Sequence[CheckResult],
    stratum: Stratum,
    weights: Mapping[str, float],
    only: frozenset[str] | None = None,
) -> AggregateScore:
    by_check: dict[str, list[CheckResult]] = {}
    for result in results:
        if weights.get(result.check_id, 0.0) > 0.0 and (only is None or result.check_id in only):
            by_check.setdefault(result.check_id, []).append(result)
    votes: list[Vote] = []
    for check_id in sorted(by_check):
        check = stratum.check(check_id)
        found = (
            None
            if check is None
            else graded_vote(api, check_id, by_check[check_id], check, weights[check_id])
        )
        if found is not None:
            votes.append(found)
    return _aggregate(api, votes, sum(weights[c] for c in by_check))


def _aggregate(
    api: CalibrationApi[Stratum], votes: Sequence[Vote], with_event: float
) -> AggregateScore:
    graded = sum(v.weight for v in votes)
    score, sigma = api.weighted_score(votes) if votes else (None, None)
    return AggregateScore(
        score=score,
        sigma=sigma,
        abstained=not votes,
        reason=None if votes else NO_GRADED,
        n_decided=len(votes),
        n_failed=sum(v.failed for v in votes),
        coverage=graded / with_event if with_event > 0.0 else 0.0,
        decided=frozenset(v.check_id for v in votes),
        per_check={v.check_id: v.p for v in votes},
    )


class GradedScorer(CalibratedGate):
    name: ClassVar[str] = "graded"
    version: ClassVar[str] = "1"
    calibrated: ClassVar[bool] = True

    def score(
        self, results: Sequence[CheckResult], only: frozenset[str] | None = None
    ) -> AggregateScore:
        api, stratum = self._opened(results)
        weights = stratum_weights(stratum, monotonicity_weight)
        return graded_score(api, results, stratum, weights, only)


PLUGINS = (GradedScorer,)
