"""The H3 verdict (ADR 0013 D8): fixed precedence, the first matching tier wins. Pure.

The external step is tested only after a supported primary (fixed sequence). Non-inferiority and
the interval arm of a contradiction use the bootstrap interval of the mean, on the scale of the
margin; superiority and the significant-negative arm use Wilcoxon + Hodges-Lehmann.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

from physverify.core.models import FrozenModel
from physverify.generator_comparison.stats import TestResult, significant_negative, supported

GuardState = Literal["ok", "fail", "not_evaluated"]
ExternalStatus = Literal["ok", "failed_gate", "unusable", "not_run"]
ExternalOutcome = Literal["confirmed", "noninferior", "inconclusive", "contradicted", "not_tested"]
RuleStatus = Literal["calibrated", "provisional", "unavailable"]

NO_VERDICT: Final = "no_verdict_provisional"
NO_FT_MODEL: Final = "no_ft_model"
HACKING: Final = "hacking_suspected"
NOT_SUPPORTED: Final = "not_supported"
CONTRADICTED: Final = "contradicted_external"
CONFIRMED: Final = "supported_external_confirmed"
NONINFERIOR: Final = "supported_external_noninferior"
VERIFIER_ONLY: Final = "supported_verifier_only"
POST_HOC_REASON: Final = "post hoc: the pre-registration changed after eval data existed"
TIERS: Final = (
    NO_VERDICT,
    NO_FT_MODEL,
    HACKING,
    NOT_SUPPORTED,
    CONTRADICTED,
    CONFIRMED,
    NONINFERIOR,
    VERIFIER_ONLY,
)


@dataclass(frozen=True)
class VerdictInputs:
    rule_status: RuleStatus
    contaminated: bool | None
    label_matches: bool
    has_ft_model: bool
    primary: TestResult | None
    external_status: ExternalStatus
    external: TestResult | None
    margin_pc: float
    alpha: float
    guards: Mapping[str, GuardState] = field(default_factory=dict)
    post_hoc: bool = False
    stop_rule_fired: str | None = None


class H3Verdict(FrozenModel):
    """``post_hoc``: an outcome-defining field was refrozen after eval data existed."""

    verdict: str
    external_status: ExternalStatus
    external_outcome: ExternalOutcome
    reasons: list[str]
    post_hoc: bool


def external_outcome(
    primary: TestResult | None,
    status: ExternalStatus,
    external: TestResult | None,
    margin: float,
    alpha: float,
) -> ExternalOutcome:
    if not supported(primary, alpha) or status != "ok" or external is None:
        return "not_tested"
    if significant_negative(external, alpha) or (
        external.ci_hi is not None and external.ci_hi < -margin
    ):
        return "contradicted"
    if supported(external, alpha):
        return "confirmed"
    if external.ci_lo is not None and external.ci_lo > -margin:
        return "noninferior"
    return "inconclusive"


def _provisional_reasons(v: VerdictInputs) -> list[str]:
    reasons = []
    if v.rule_status != "calibrated":
        reasons.append(f"rule status {v.rule_status}")
    if v.contaminated is None:
        reasons.append("calibration dataset provenance is unknown")
    elif v.contaminated:
        reasons.append("calibration dataset is contaminated")
    if not v.label_matches:
        reasons.append("score stamp differs from the frozen rule label")
    if v.stop_rule_fired is not None:
        reasons.append(v.stop_rule_fired)
    return reasons


OUTCOME_TIER: Final[Mapping[ExternalOutcome, str]] = {
    "contradicted": CONTRADICTED,
    "confirmed": CONFIRMED,
    "noninferior": NONINFERIOR,
    "inconclusive": VERIFIER_ONLY,
    "not_tested": VERIFIER_ONLY,
}


def _tier(v: VerdictInputs, outcome: ExternalOutcome) -> tuple[str, list[str]]:
    provisional = _provisional_reasons(v)
    if provisional:
        return NO_VERDICT, provisional
    if not v.has_ft_model:
        return NO_FT_MODEL, ["no fine-tuned checkpoint passed selection"]
    failed = sorted(name for name, state in v.guards.items() if state == "fail")
    if failed:
        return HACKING, [f"guard {name} failed" for name in failed]
    if not supported(v.primary, v.alpha):
        return NOT_SUPPORTED, ["primary test not significant with HL > 0"]
    return OUTCOME_TIER[outcome], [f"external {v.external_status}: {outcome}"]


def h3_verdict(v: VerdictInputs) -> H3Verdict:
    outcome = external_outcome(v.primary, v.external_status, v.external, v.margin_pc, v.alpha)
    verdict, reasons = _tier(v, outcome)
    late = [POST_HOC_REASON] if v.post_hoc else []
    return H3Verdict(
        verdict=verdict,
        external_status=v.external_status,
        external_outcome=outcome,
        reasons=[*reasons, *late],
        post_hoc=v.post_hoc,
    )
