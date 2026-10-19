"""Provisional score from the verifier's verdicts and thresholds: for plumbing and tests only.

Per decided check j (FAIL in any subject, or PASS in all), the worst subject gives
``rho_j = max(0, m_j / tau_j)``, unclipped, and ``s_j = 1 / (1 + rho_j)``: a PASS has
``rho_j <= 1`` so ``s_j >= 0.5``; a FAIL has ``rho_j > 1`` so ``s_j < 0.5``. Over the n decided
checks (or those in ``only``) the score is their geometric mean,
``score = exp(-(1/n) sum_j ln(1 + rho_j))`` in (0, 1], strictly decreasing in every ``rho_j``,
so two failing candidates still rank by how badly they fail. Linear propagation gives
``sigma = (score / n) sqrt(sum_j ((sigma_j / tau_j) / (1 + rho_j))^2)``; a term clamped at
``rho_j = 0`` contributes 0 (the clamp is flat there) and any other unknown ``sigma_j`` makes
sigma None. It is not a calibrated probability (ADR 0012, amendment 1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from physverify.config.models import Settings
from physverify.finetune.scorers._base import AggregateScore
from physverify.schema.checks import CheckResult, Verdict
from physverify.verification.decisive import decisive
from physverify.verification.runner import (
    VERIFICATION_SECTION,
    VerificationSettings,
    load_thresholds,
    thresholds_label,
)

NO_DECIDED = "no_decided_check"
DECIDED_VERDICTS = frozenset({Verdict.PASS, Verdict.FAIL})


@dataclass(frozen=True)
class CheckTerm:
    """``ratio`` is ``rho_j >= 0`` (unbounded); ``sigma`` is the sigma of ``rho_j``."""

    check_id: str
    failed: bool
    ratio: float
    sigma: float | None

    @property
    def factor(self) -> float:
        return 1.0 / (1.0 + self.ratio)


def check_verdict(results: Sequence[CheckResult]) -> Verdict | None:
    """FAIL if any subject fails, PASS if all pass, else None (not decided)."""
    verdicts = {result.verdict for result in results}
    if Verdict.FAIL in verdicts:
        return Verdict.FAIL
    return Verdict.PASS if verdicts == {Verdict.PASS} else None


def _term(check_id: str, results: Sequence[CheckResult]) -> CheckTerm | None:
    verdict = check_verdict(results)
    scored = [
        r
        for r in results
        if r.verdict in DECIDED_VERDICTS and r.magnitude is not None and r.threshold
    ]
    worst = decisive(scored)
    if verdict is None or worst is None:
        return None
    tau = float(worst.threshold or 1.0)
    raw = float(worst.magnitude or 0.0) / tau
    if raw < 0.0:
        sigma: float | None = 0.0
    else:
        sigma = worst.sigma / tau if math.isfinite(worst.sigma) else None
    return CheckTerm(check_id, verdict is Verdict.FAIL, max(0.0, raw), sigma)


def group_by_check(results: Sequence[CheckResult]) -> dict[str, list[CheckResult]]:
    grouped: dict[str, list[CheckResult]] = {}
    for result in results:
        grouped.setdefault(result.check_id, []).append(result)
    return grouped


def geometric_score(terms: Sequence[CheckTerm]) -> float:
    return math.exp(-math.fsum(math.log1p(term.ratio) for term in terms) / len(terms))


def propagated_sigma(terms: Sequence[CheckTerm], score: float) -> float | None:
    if any(term.sigma is None for term in terms):
        return None
    return score / len(terms) * math.hypot(*((term.sigma or 0.0) * term.factor for term in terms))


def provisional_score(
    results: Sequence[CheckResult], only: frozenset[str] | None = None
) -> AggregateScore:
    grouped = {
        check_id: rows
        for check_id, rows in group_by_check(results).items()
        if (only is None or check_id in only)
        and any(r.verdict is not Verdict.UNSCORED for r in rows)
    }
    terms = [t for t in (_term(c, rows) for c, rows in sorted(grouped.items())) if t is not None]
    coverage = len(terms) / len(grouped) if grouped else 0.0
    if not terms:
        return AggregateScore(None, None, True, NO_DECIDED, 0, 0, coverage)
    score = geometric_score(terms)
    return AggregateScore(
        score=score,
        sigma=propagated_sigma(terms, score),
        abstained=False,
        reason=None,
        n_decided=len(terms),
        n_failed=sum(term.failed for term in terms),
        coverage=coverage,
        decided=frozenset(term.check_id for term in terms),
        per_check={term.check_id: term.factor for term in terms},
    )


class ProvisionalScorer:
    name: ClassVar[str] = "provisional"
    version: ClassVar[str] = "2"
    calibrated: ClassVar[bool] = False

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def unavailable(self) -> str | None:
        return None

    def thresholds(self) -> str:
        verification = self._settings.section(VERIFICATION_SECTION, VerificationSettings)
        table = load_thresholds(self._settings.paths.thresholds, verification.thresholds)
        return thresholds_label(table)

    def score(
        self, results: Sequence[CheckResult], only: frozenset[str] | None = None
    ) -> AggregateScore:
        return provisional_score(results, only)


PLUGINS = (ProvisionalScorer,)
