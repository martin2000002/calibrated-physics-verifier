"""One clip verdict for fit and apply (0b.1): multi-subject clips, a missing input, parity."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
from calibration_cases import (
    CHECKS,
    check_calibration,
    config,
    fit_meta,
    missing,
    result,
    row,
    synthetic_report,
    view,
)

from physverify.calibration.apply import (
    check_outcome,
    policy_for,
    rule_for,
    score_clip,
)
from physverify.calibration.artifact_io import Stratum as StratumView
from physverify.calibration.collect import level_data
from physverify.calibration.fit import FitInputs, fit_calibration, platt_data
from physverify.calibration.table import CalibrationDataError, Role, results_by_row
from physverify.calibration.thresholds import tpr_by_level, verdict_of
from physverify.calibration.verdict import DECIDED, NO_EVENT, Rule, clip_verdict
from physverify.schema.checks import CheckResult, InputLevel, Verdict
from physverify.schema.clip import Split
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.runner import RowStatus, VerificationReport, aggregate

RULE = Rule(ThresholdPolicy(), bias=0.0, factor=1.0)
TAU = 0.08


def abstaining_larger_and_failing_smaller() -> list[CheckResult]:
    """A: m 0.10, sigma 0.5 crosses tau; B: m 0.09, sigma 0 fails (the review's case a)."""
    return [
        result("R1", "c", 0.10, 0.5, InputLevel.ORACLE, subject="a"),
        result("R1", "c", 0.09, 0.0, InputLevel.ORACLE, subject="b"),
    ]


def test_a_failing_subject_wins_over_a_larger_abstaining_one() -> None:
    outcome = clip_verdict(abstaining_larger_and_failing_smaller(), RULE, TAU)
    assert outcome.verdict is Verdict.FAIL and outcome.magnitude == 0.09


def test_fit_and_apply_both_fail_the_multi_subject_clip() -> None:
    subjects = abstaining_larger_and_failing_smaller()
    positive = row(0.10, role=Role.POSITIVE, level=1, mu=1.0, sigma=0.5, subjects=subjects)
    (level,) = tpr_by_level([positive], RULE, TAU, 0.95)
    assert level.total.k == 1
    stratum = view({"R1": check_calibration(tau=TAU)})
    outcome = check_outcome(subjects, stratum, policy_for(stratum))
    assert outcome is not None and outcome.verdict is Verdict.FAIL
    assert score_clip(subjects, stratum).n_failed == 1


def test_a_missing_input_makes_a_passing_check_abstain_in_fit_and_apply() -> None:
    """The review's case b: one subject passes, the other has no input."""
    subjects = [result("R1", "c", 0.01, 0.0, InputLevel.ORACLE), missing("R1", "c")]
    clean = row(0.01, subjects=subjects)
    assert verdict_of(clean, RULE, TAU) is Verdict.ABSTAIN
    stratum = view({"R1": check_calibration(tau=TAU)})
    clip = score_clip(subjects, stratum)
    assert clip.abstained_checks == ("R1",) and clip.decided_checks == ()


def test_platt_is_fitted_on_decided_outcomes_only() -> None:
    """A clip whose sigma straddles tau abstains in ``score_clip``, so it gets no p_c in the fit."""
    straddling = result("R1", "c", 0.10, 0.5, InputLevel.ORACLE)
    undecided = row(0.10, sigma=0.5, subjects=[straddling])
    decided = row(0.20, scene="s1")
    assert [d.magnitude for d in platt_data([undecided, decided], RULE, TAU)] == [0.20]


def test_a_scored_row_without_its_subject_results_is_rejected() -> None:
    report = synthetic_report(20)
    target = next(r for r in report.table if r.status is RowStatus.SCORED)
    kept = tuple(
        r for r in report.results if (r.clip_id, r.check_id) != (target.clip_id, target.check_id)
    )
    broken = VerificationReport(results=kept, table=report.table, clips=report.clips)
    with pytest.raises(CalibrationDataError, match="no per-subject results"):
        level_data(InputLevel.ORACLE, broken, CHECKS, config(), split=Split.CALIB)


def test_no_tau_or_no_event_leaves_the_check_unscored() -> None:
    assert clip_verdict([result("R1", "c", 0.5)], RULE, None).verdict is Verdict.UNSCORED
    assert clip_verdict([], RULE, TAU) is NO_EVENT


def _subjects(rng: np.random.Generator, first: CheckResult) -> list[CheckResult]:
    """The fixture's subject plus a second one: smaller, noisier, or without input."""
    sigma = float(rng.choice([0.0, 0.005, 0.05]))
    found = [result(first.check_id, first.clip_id, first.magnitude, sigma, first.input_level)]
    draw = rng.random()
    if draw < 0.2:
        found.append(missing(first.check_id, first.clip_id, first.input_level))
    elif draw < 0.8:
        scale = float(rng.uniform(0.3, 1.3))
        noise = float(rng.choice([0.0, 0.01, 0.1]))
        found.append(
            result(
                first.check_id,
                first.clip_id,
                (first.magnitude or 0.0) * scale,
                noise,
                first.input_level,
                subject="b",
            )
        )
    return found


def multi_subject_report(n_scenes: int, seed: int) -> VerificationReport:
    base = synthetic_report(n_scenes, seed)
    rng = np.random.default_rng(seed + 1)
    results = [s for r in base.results for s in _subjects(rng, r)]
    grouped = results_by_row(results)
    units = {c.check_id: c for c in CHECKS}
    table = tuple(
        aggregate(
            row.clip_id,
            cast(Any, units[row.check_id]),
            grouped.get((row.clip_id, row.check_id), ()),
            row_step=row.row_step,
        )
        for row in base.table
    )
    return VerificationReport(results=tuple(results), table=table, clips=base.clips)


def _discrepancies(stratum: StratumView, rows: Sequence[Any]) -> list[str]:
    policy = policy_for(stratum)
    found: list[str] = []
    for r in rows:
        tau = stratum.tau(r.check_id)
        if tau is None or not r.subjects:
            continue
        rule = rule_for(stratum, r.check_id, policy)
        applied = check_outcome(r.subjects, stratum, policy)
        assert applied is not None
        fitted = verdict_of(r, rule, tau)
        if fitted is not applied.verdict:
            found.append(f"{r.clip_id}/{r.check_id}: fit {fitted} apply {applied.verdict}")
        if applied.verdict in DECIDED and r.scored:
            (platt,) = platt_data([r], rule, tau) or [None]
            if platt is None or platt.magnitude != applied.magnitude:
                found.append(f"{r.clip_id}/{r.check_id}: Platt magnitude differs")
    return found


def test_fit_and_score_clip_agree_on_every_multi_subject_clip() -> None:
    """Parity 100 %: the fit-time verdict and magnitude of each (clip, check) equal the outcome
    ``score_clip`` applies, and the decided/abstained split of ``score_clip`` follows it."""
    report = multi_subject_report(40, seed=3)
    data = level_data(InputLevel.ORACLE, report, CHECKS, config(), split=Split.CALIB)
    assert sum(len(r.subjects) > 1 for r in data.rows) > 100
    artifact = fit_calibration(FitInputs((data,), CHECKS, fit_meta()), config())
    (model,) = artifact.strata
    stratum = StratumView(model=model, source="fit", artifact=None)
    verdicts = [
        verdict_of(r, rule_for(stratum, r.check_id, policy_for(stratum)), stratum.tau("R1"))
        for r in data.rows
        if r.check_id == "R1"
    ]
    assert {Verdict.PASS, Verdict.FAIL, Verdict.ABSTAIN} <= set(verdicts)
    assert _discrepancies(stratum, data.rows) == []
    rows = {(r.clip_id, r.check_id): r for r in data.rows}
    for clip_id, results in sorted(_by_clip(report.results).items()):
        clip = score_clip(results, stratum)
        for check_id in (c for c in stratum.model.checks if stratum.tau(c) is not None):
            r = rows[(clip_id, check_id)]
            rule = rule_for(stratum, check_id, policy_for(stratum))
            decided = r.subjects and verdict_of(r, rule, stratum.tau(check_id)) in DECIDED
            assert (check_id in clip.decided_checks) is bool(decided)


def _by_clip(results: Sequence[CheckResult]) -> dict[str, list[CheckResult]]:
    found: dict[str, list[CheckResult]] = {}
    for r in results:
        found.setdefault(r.clip_id, []).append(r)
    return found


BIASED = Rule(ThresholdPolicy(), bias=0.02, factor=1.5)
BIASED_THRESHOLDS = {"checks": {"R1": {"tau": TAU, "bias": 0.02, "sigma_factor": 1.5}}}
NAN = float("nan")


def _random_subjects(rng: np.random.Generator) -> list[CheckResult]:
    found: list[CheckResult] = []
    for index in range(int(rng.integers(1, 5))):
        draw = rng.random()
        if draw < 0.15:
            found.append(missing("R1", subject=f"s{index}"))
            continue
        magnitude = NAN if draw < 0.3 else float(rng.uniform(0.0, 0.25))
        sigma = float(rng.choice([0.0, 0.005, 0.05, NAN]))
        found.append(result("R1", "c", magnitude, sigma, InputLevel.ORACLE, subject=f"s{index}"))
    return found


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b)


@pytest.mark.parametrize(
    ("thresholds", "tau", "expected"),
    [
        (BIASED_THRESHOLDS, TAU, {Verdict.PASS, Verdict.FAIL, Verdict.ABSTAIN}),
        ({}, None, {Verdict.UNSCORED, Verdict.ABSTAIN}),
    ],
)
def test_clip_verdict_matches_the_run_table_row(
    thresholds: dict[str, Any], tau: float | None, expected: set[Verdict]
) -> None:
    """``clip_verdict`` and ``aggregate`` agree on the verdict and on magnitude (m - bias + bias)
    for random sets with NaN magnitudes, unknown sigmas and missing inputs."""
    rng = np.random.default_rng(11)
    policy = ThresholdPolicy()
    unit = cast(Any, next(c for c in CHECKS if c.check_id == "R1"))
    seen: set[Verdict] = set()
    for _ in range(400):
        subjects = _random_subjects(rng)
        table_row = aggregate("c", unit, [policy.decide(s, thresholds) for s in subjects])
        outcome = clip_verdict(subjects, BIASED, tau)
        magnitude = None if outcome.magnitude is None else outcome.magnitude + BIASED.bias
        assert outcome.verdict is table_row.verdict, (subjects, outcome, table_row)
        assert _same(magnitude, table_row.magnitude), (subjects, outcome, table_row)
        seen.add(outcome.verdict)
    assert expected <= seen


def test_a_pass_and_an_abstain_clip_abstains_with_the_pass_magnitude() -> None:
    subjects = [
        result("R1", "c", 0.10, 0.5, InputLevel.ORACLE, subject="a"),
        result("R1", "c", 0.01, 0.0, InputLevel.ORACLE, subject="b"),
    ]
    outcome = clip_verdict(subjects, RULE, TAU)
    assert outcome.verdict is Verdict.ABSTAIN and outcome.magnitude == 0.01
    assert outcome.abstain_reason is not None and not outcome.judged
    stratum = view({"R1": check_calibration(tau=TAU)})
    applied = check_outcome(subjects, stratum, policy_for(stratum))
    assert applied is not None and applied.verdict is Verdict.ABSTAIN
    assert applied.p_violation is None


def test_a_fail_and_a_nan_clip_fails() -> None:
    subjects = [
        result("R1", "c", NAN, 0.0, InputLevel.ORACLE, subject="a"),
        result("R1", "c", 0.2, 0.0, InputLevel.ORACLE, subject="b"),
    ]
    outcome = clip_verdict(subjects, RULE, TAU)
    assert outcome.verdict is Verdict.FAIL and outcome.magnitude == 0.2


def test_without_tau_an_unscored_and_a_larger_abstained_subject_give_unscored() -> None:
    """Behaviour change: the clip follows the run table (UNSCORED if any is), not the verdict of
    the largest undecided subject, and reports the unscored subject's magnitude."""
    subjects = [
        result("R1", "c", 0.02, 0.0, InputLevel.ORACLE, subject="a"),
        replace(missing("R1", "c", InputLevel.ORACLE), magnitude=0.5),
    ]
    outcome = clip_verdict(subjects, RULE, None)
    assert outcome.verdict is Verdict.UNSCORED and outcome.magnitude == 0.02
    assert outcome.abstain_reason is None and not outcome.judged
