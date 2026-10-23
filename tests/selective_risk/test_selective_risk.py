"""ADR 0010 I6-I7: the risk-coverage sweep reuses the threshold policy and its statistics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy.stats import binomtest, norm

from physverify.app.plugins import PLUGIN_KINDS
from physverify.calibration.table import Role
from physverify.diagnosis.pairing import Paired, ResultPair
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.selective_risk.curve import (
    ABSTAINED,
    FAILED,
    PASSED,
    Case,
    CaseCounts,
    alpha_grid,
    aurc,
    curve_point,
    decision_codes,
    injected_reference,
    k_of,
    paired_cases,
    point_reference,
    raw_table,
    reference_codes,
    risk_bounds,
    row_codes,
)
from physverify.selective_risk.fit import choose_k, paired_aurc_difference, z_scores
from physverify.selective_risk.settings import AbstentionSettings
from physverify.verification.runner import VerificationSettings, build_policy

CODE = {Verdict.PASS: PASSED, Verdict.FAIL: FAILED}


def result(
    magnitude: float | None,
    sigma: float,
    *,
    check_id: str = "R1",
    clip_id: str = "c0",
    reason: AbstainReason | None = None,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        clip_id=clip_id,
        subject=("a",),
        frame_span=(0, 3),
        magnitude=magnitude,
        sigma=sigma,
        unit="",
        verdict=Verdict.UNSCORED if reason is None else Verdict.ABSTAIN,
        input_level=InputLevel.PERCEIVED,
        abstain_reason=reason,
    )


def table(**entries: dict[str, Any]) -> dict[str, Any]:
    return {"status": "calibrated", "source": "test", "checks": entries}


def random_cases(n: int, seed: int) -> tuple[list[Case], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    thresholds = table(
        R1={"tau": 0.05},
        R2={"tau": 0.01, "bias": 0.004, "sigma_factor": 1.7},
        R3a={"tau": 0.1, "sigma_factor": None},
    )
    ids = ["R1", "R2", "R3a", "R9"]
    cases = []
    for i in range(n):
        check_id = ids[i % len(ids)]
        sigma = float(rng.choice([0.0, math.nan, *rng.uniform(0.001, 0.1, 6)]))
        perceived = result(
            float(rng.normal(0.05, 0.08)), sigma, check_id=check_id, clip_id=f"c{i % 37}"
        )
        reference = Verdict.PASS if rng.random() < 0.6 else Verdict.FAIL
        cases.append(Case(perceived.clip_id, check_id, reference, perceived))
    return cases, thresholds


@pytest.mark.parametrize("k", [1.0, 2.0, 3.0])
def test_the_sweep_reproduces_verify_result_by_result(k: float) -> None:
    cases, thresholds = random_cases(10_000, seed=int(k))
    settings = VerificationSettings(policies={"threshold": {"k_sigma": k}})
    policy = build_policy(PLUGIN_KINDS["policies"](), settings)
    expected = [
        CODE.get(policy.decide(c.perceived, thresholds).verdict, ABSTAINED)
        for c in cases
        if c.perceived is not None
    ]
    assert decision_codes(cases, [k], thresholds)[0].tolist() == expected


def test_a_missed_event_and_a_check_abstention_count_as_abstained() -> None:
    thresholds = table(R1={"tau": 0.05})
    cases = [
        Case("c", "R1", Verdict.PASS, None),
        Case("c", "R1", Verdict.PASS, result(None, math.nan, reason=AbstainReason.OCCLUDED)),
        Case("c", "R1", Verdict.FAIL, result(0.9, 0.01)),
    ]
    assert decision_codes(cases, [2.0], thresholds)[0].tolist() == [ABSTAINED, ABSTAINED, FAILED]


def test_coverage_and_risk_on_a_hand_made_set() -> None:
    codes = np.array([PASSED, PASSED, FAILED, ABSTAINED, FAILED, PASSED])
    reference = np.array([PASSED, FAILED, FAILED, PASSED, PASSED, PASSED])
    point = curve_point(codes, reference)
    assert (point.n, point.decided, point.errors) == (6, 5, 2)
    assert point.coverage == pytest.approx(5 / 6) and point.risk == pytest.approx(2 / 5)
    assert (point.false_fail, point.ref_pass_decided) == (1, 3)
    assert (point.missed_fail, point.ref_fail_decided) == (1, 2)


@pytest.mark.parametrize(("x", "n"), [(0, 10), (3, 10), (10, 10), (1, 250), (17, 40)])
def test_risk_bounds_are_the_exact_binomial_interval(x: int, n: int) -> None:
    exact = binomtest(x, n).proportion_ci(confidence_level=0.95, method="exact")
    low, high = risk_bounds(x, n, 0.95)
    assert abs(low - exact.low) < 1e-9 and abs(high - exact.high) < 1e-9


def test_risk_bounds_without_decisions_are_the_whole_unit_interval() -> None:
    assert risk_bounds(0, 0, 0.95) == (0.0, 1.0)


def test_coverage_does_not_grow_as_alpha_shrinks() -> None:
    cases, thresholds = random_cases(2_000, seed=7)
    ks = k_of(alpha_grid(AbstentionSettings()))
    assert np.all(np.diff(ks) > 0)
    codes = decision_codes(cases, ks, thresholds)
    coverage = [curve_point(row, reference_codes(cases)).coverage for row in codes]
    assert all(b <= a for a, b in zip(coverage, coverage[1:], strict=False))
    assert ks[0] == pytest.approx(norm.isf(0.5)) and ks[-1] == pytest.approx(norm.isf(1e-4))


def test_k_star_is_none_when_no_point_meets_the_risk_target() -> None:
    reference = np.array([PASSED] * 10)
    noisy = np.array([[FAILED] * 5 + [PASSED] * 5, [ABSTAINED] * 9 + [FAILED]])
    points = [curve_point(row, reference) for row in noisy]
    assert choose_k(points, [1.0, 2.0], risk_target=0.1, ci=0.95) is None


def test_k_star_takes_the_largest_coverage_then_the_smallest_k() -> None:
    reference = np.array([PASSED] * 60)
    rows = np.array([[PASSED] * 60, [PASSED] * 60, [PASSED] * 30 + [ABSTAINED] * 30])
    points = [curve_point(row, reference) for row in rows]
    assert choose_k(points, [2.0, 1.0, 3.0], risk_target=0.1, ci=0.95) == 1


def test_aurc_is_the_mean_risk_over_the_coverage_span() -> None:
    assert aurc([0.2, 0.6, 1.0], [0.0, 0.1, 0.3]) == pytest.approx((0.02 + 0.08) / 0.8)
    assert aurc([0.5], [0.2]) == pytest.approx(0.2)
    assert math.isnan(aurc([0.0], [math.nan]))


def test_the_paired_bootstrap_of_identical_tables_is_zero() -> None:
    cases, thresholds = random_cases(300, seed=1)
    codes = decision_codes(cases, [0.5, 1.0, 2.0], thresholds)
    clips = [c.clip_id for c in cases]
    diff = paired_aurc_difference(
        codes, codes, reference_codes(cases), clips, n_boot=30, seed=0, ci=0.95
    )
    assert diff["value"] == 0.0 and diff["ci"] == (0.0, 0.0)


def test_the_reference_is_the_point_verdict_of_the_oracle_without_correction() -> None:
    reference = raw_table(table(R1={"tau": 0.05, "bias": 0.01, "sigma_factor": None}))
    assert reference["checks"] == {"R1": {"tau": 0.05}}
    assert point_reference(result(0.055, 0.5), reference) is Verdict.FAIL
    assert point_reference(result(0.045, math.nan), reference) is Verdict.PASS
    assert point_reference(result(0.07, 0.5, check_id="R9"), reference) is None


def test_pairs_become_cases_and_misses_stay_as_abstentions() -> None:
    thresholds = table(R1={"tau": 0.05, "bias": 0.04})
    hit = ResultPair("R1", "c0", ("a",), result(0.06, 0.0), result(0.2, 0.01))
    paired = Paired(
        pairs=(hit,),
        missing=(result(0.01, 0.0), result(0.3, 0.0, check_id="R9")),
        spurious=(result(0.5, 0.01),),
    )
    cases, counts = paired_cases(paired, thresholds)
    assert [(c.reference, c.perceived) for c in cases] == [
        (Verdict.FAIL, hit.other),
        (Verdict.PASS, None),
    ]
    assert counts == CaseCounts(no_reference=1, spurious=1)


@pytest.mark.parametrize(
    ("role", "verdict"),
    [
        (Role.CLEAN, Verdict.PASS),
        (Role.POSITIVE, Verdict.FAIL),
        (Role.COUPLED, None),
        (Role.OFF_TARGET, None),
    ],
)
def test_the_injected_label_follows_the_calibration_role(role: Role, verdict: Verdict) -> None:
    assert injected_reference(role) is verdict


def test_z_is_computed_for_the_report_only() -> None:
    thresholds = table(R1={"tau": 0.05, "bias": 0.01, "sigma_factor": 2.0})
    perceived = result(0.1, 0.01)
    z = z_scores([Case("c", "R1", Verdict.FAIL, perceived)], thresholds)
    assert z == [pytest.approx((0.1 - 0.01 - 0.05) / 0.02)]
    assert perceived.details == {}


def test_rows_follow_the_table_rule_of_verify() -> None:
    keys = [("c", "R1"), ("c", "R1"), ("c", "R2"), ("c", "R2"), ("d", "R1")]
    codes = np.array([PASSED, FAILED, PASSED, ABSTAINED, PASSED])
    assert row_codes(keys, codes) == {
        ("c", "R1"): FAILED,
        ("c", "R2"): ABSTAINED,
        ("d", "R1"): PASSED,
    }
