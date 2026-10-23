from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import pytest

from physverify.finetune.stop_rule import StopRuleState, fired_reason
from physverify.generator_comparison.stats import TestResult
from physverify.generator_comparison.verdict import (
    CONFIRMED,
    CONTRADICTED,
    HACKING,
    NO_FT_MODEL,
    NO_VERDICT,
    NONINFERIOR,
    NOT_SUPPORTED,
    VERIFIER_ONLY,
    ExternalStatus,
    GuardState,
    RuleStatus,
    VerdictInputs,
    comparison_verdict,
)

MARGIN = 0.13


def _test(p: float | None, hl: float | None, lo: float | None, hi: float | None) -> TestResult:
    return TestResult(n=20, p=p, hl=hl, mean=hl, ci_lo=lo, ci_hi=hi, method="m")


SUPPORTED = _test(0.001, 0.1, 0.05, 0.15)
NULL = _test(0.6, 0.0, -0.05, 0.05)


def _inputs(**update: Any) -> VerdictInputs:
    base = VerdictInputs(
        rule_status="calibrated",
        contaminated=False,
        label_matches=True,
        has_ft_model=True,
        primary=SUPPORTED,
        external_status="ok",
        external=_test(0.01, 0.3, 0.1, 0.5),
        margin_pc=MARGIN,
        alpha=0.05,
        guards={"G1": "ok", "G4": "not_evaluated"},
    )
    return dataclasses.replace(base, **update)


@pytest.mark.parametrize(
    "update",
    [
        {"rule_status": "provisional"},
        {"rule_status": "unavailable"},
        {"contaminated": True},
        {"contaminated": None},
        {"label_matches": False},
    ],
)
def test_a_provisional_rule_never_gives_a_verdict(update: dict[str, Any]) -> None:
    result = comparison_verdict(_inputs(**update, has_ft_model=False, guards={"G1": "fail"}))
    assert result.verdict == NO_VERDICT


def test_no_fine_tuned_model_comes_before_the_guards() -> None:
    result = comparison_verdict(_inputs(has_ft_model=False, guards={"G1": "fail"}))
    assert result.verdict == NO_FT_MODEL


def test_a_failed_guard_is_hacking() -> None:
    result = comparison_verdict(_inputs(guards={"G2": "fail", "G1": "ok", "G5": "fail"}))
    assert result.verdict == HACKING
    assert result.reasons == ["guard G2 failed", "guard G5 failed"]


def test_the_external_judge_is_tested_only_after_the_primary() -> None:
    result = comparison_verdict(_inputs(primary=NULL))
    assert (result.verdict, result.external_outcome) == (NOT_SUPPORTED, "not_tested")


@pytest.mark.parametrize(
    ("external", "verdict", "outcome"),
    [
        (_test(0.01, 0.3, 0.1, 0.5), CONFIRMED, "confirmed"),
        (_test(0.4, 0.0, -0.1, 0.1), NONINFERIOR, "noninferior"),
        (_test(0.4, -0.05, -0.2, 0.1), VERIFIER_ONLY, "inconclusive"),
        (_test(0.3, -0.2, -0.4, -0.14), CONTRADICTED, "contradicted"),
        (_test(0.01, -0.05, -0.1, -0.01), CONTRADICTED, "contradicted"),
    ],
)
def test_external_outcomes_use_the_mean_interval_against_the_margin(
    external: TestResult, verdict: str, outcome: str
) -> None:
    result = comparison_verdict(_inputs(external=external))
    assert (result.verdict, result.external_outcome) == (verdict, outcome)


def test_an_interval_touching_the_margin_is_not_noninferior() -> None:
    result = comparison_verdict(_inputs(external=_test(0.4, 0.0, -MARGIN, 0.1)))
    assert result.external_outcome == "inconclusive"


@pytest.mark.parametrize("status", ["failed_gate", "unusable", "not_run"])
def test_without_a_usable_judge_the_result_is_verifier_only(status: str) -> None:
    result = comparison_verdict(_inputs(external_status=status))
    assert (result.verdict, result.external_outcome) == (VERIFIER_ONLY, "not_tested")
    assert result.external_status == status


PRIMARIES: dict[str, TestResult | None] = {
    "positive": SUPPORTED,
    "negative": _test(0.001, -0.1, -0.15, -0.05),
    "null": NULL,
    "missing": None,
}
EXTERNALS: dict[str, TestResult | None] = {
    "confirmed": _test(0.01, 0.3, 0.1, 0.5),
    "noninferior": _test(0.4, 0.0, -0.1, 0.1),
    "inconclusive": _test(0.4, -0.05, -0.2, 0.1),
    "contradicted_interval": _test(0.3, -0.2, -0.4, -0.14),
    "contradicted_significant": _test(0.01, -0.05, -0.1, -0.01),
    "missing": None,
}
EXTERNAL_TIER = {
    "confirmed": CONFIRMED,
    "noninferior": NONINFERIOR,
    "inconclusive": VERIFIER_ONLY,
    "contradicted_interval": CONTRADICTED,
    "contradicted_significant": CONTRADICTED,
    "missing": VERIFIER_ONLY,
}
RULE_STATUSES: tuple[RuleStatus, ...] = ("calibrated", "provisional", "unavailable")
GUARD_STATES: tuple[GuardState, ...] = ("ok", "fail", "not_evaluated")
EXTERNAL_STATUSES: tuple[ExternalStatus, ...] = ("ok", "failed_gate", "unusable", "not_run")


def _expected(v: VerdictInputs, primary: str, external: str) -> str:
    """The ADR 0013 D8 precedence written out independently of ``verdict.py``."""
    if v.rule_status != "calibrated" or v.contaminated or not v.label_matches:
        return NO_VERDICT
    if not v.has_ft_model:
        return NO_FT_MODEL
    if "fail" in v.guards.values():
        return HACKING
    if primary != "positive":
        return NOT_SUPPORTED
    if v.external_status != "ok":
        return VERIFIER_ONLY
    return EXTERNAL_TIER[external]


@pytest.mark.parametrize(
    ("rule_status", "contaminated", "label_matches", "has_ft_model"),
    list(itertools.product(RULE_STATUSES, [False, True], [True, False], [True, False])),
)
def test_the_verdict_product_table(
    rule_status: RuleStatus, contaminated: bool, label_matches: bool, has_ft_model: bool
) -> None:
    for guard, primary, status, external in itertools.product(
        GUARD_STATES, PRIMARIES, EXTERNAL_STATUSES, EXTERNALS
    ):
        v = _inputs(
            rule_status=rule_status,
            contaminated=contaminated,
            label_matches=label_matches,
            has_ft_model=has_ft_model,
            guards={"G1": "ok", "G2": guard},
            primary=PRIMARIES[primary],
            external_status=status,
            external=EXTERNALS[external],
        )
        result = comparison_verdict(v)
        case = (guard, primary, status, external)
        assert result.verdict == _expected(v, primary, external), case
        assert result.external_status == status
        tested = primary == "positive" and status == "ok" and external != "missing"
        assert (result.external_outcome != "not_tested") is tested, case
        if tested and external.startswith("contradicted"):
            assert result.external_outcome == "contradicted", case
            assert result.verdict != VERIFIER_ONLY, case


def test_unknown_provenance_is_named_and_never_clean() -> None:
    result = comparison_verdict(_inputs(contaminated=None))
    assert result.verdict == NO_VERDICT
    assert "calibration dataset provenance is unknown" in result.reasons


def test_a_post_hoc_freeze_marks_the_verdict_without_changing_its_tier() -> None:
    plain, late = comparison_verdict(_inputs()), comparison_verdict(_inputs(post_hoc=True))
    assert not plain.post_hoc and late.post_hoc
    assert late.verdict == plain.verdict
    assert late.reasons[-1].startswith("post hoc")


def test_a_fired_stop_rule_leaves_h3_without_a_verdict() -> None:
    state = StopRuleState(passed=False, enforced=True, reasons=["few pairs"], pairs_sha256="p")
    reason = fired_reason(state, "pilot accepted")
    assert reason is not None and "'pilot accepted'" in reason
    result = comparison_verdict(_inputs(stop_rule_fired=reason))
    assert result.verdict == NO_VERDICT and reason in result.reasons
    assert fired_reason(state.model_copy(update={"passed": True}), None) is None
