"""L_C conformal guarantee (AC6), honest FPR and the L_D fit with censoring (AC7, Juicio 3)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from calibration_cases import result, row, settings

from physverify.calibration.fit import check_status
from physverify.calibration.schema import UNRANKED_NEGATIVES
from physverify.calibration.table import CalibrationDataError, Role, require_subjects
from physverify.calibration.thresholds import (
    NOT_DETERMINED,
    Detection,
    cv_verdicts,
    detection_limit,
    fails,
    false_positive_rates,
    fit_tau,
    fpr_summary,
    tpr_by_level,
)
from physverify.calibration.verdict import Rule
from physverify.schema.checks import InputLevel, Verdict
from physverify.verification.policies.threshold import ThresholdPolicy

ALPHA = 0.05
RULE = Rule(ThresholdPolicy(), bias=0.0, factor=1.0)


def clean_rows(values: np.ndarray) -> list:  # type: ignore[type-arg]
    return [row(float(v), scene=f"s{i}") for i, v in enumerate(values)]


def test_the_conformal_tau_keeps_the_clean_fpr_below_alpha() -> None:
    """AC6: 1000 replicates of 100 clean scenes; FPR on fresh clean scenes <= alpha + 0.01."""
    rng = np.random.default_rng(0)
    rates = []
    for _ in range(1000):
        tau = fit_tau(clean_rows(rng.normal(0.0, 1.0, 100)), 0.0, ALPHA, 0.0)
        assert tau.sufficient and tau.value is not None
        rates.append(float(np.mean(rng.normal(0.0, 1.0, 200) > tau.value)))
    assert float(np.mean(rates)) <= ALPHA + 0.01


def test_too_few_clean_scenes_is_insufficient() -> None:
    short = fit_tau(clean_rows(np.linspace(0.0, 1.0, 18)), 0.0, ALPHA, 0.0)
    assert not short.sufficient and short.n == 18
    assert check_status(short, [row(0.5, role=Role.POSITIVE)]) == "insufficient"
    none = fit_tau([], 0.0, ALPHA, 0.0)
    assert check_status(none, []) == "no_negatives"
    full = fit_tau(clean_rows(np.linspace(0.0, 1.0, 19)), 0.0, ALPHA, 0.0)
    assert check_status(full, []) == "no_positives"
    assert check_status(full, [row(0.5, role=Role.POSITIVE)]) == "ok"


def test_a_nan_subject_does_not_hide_a_larger_one_from_tau() -> None:
    """Subjects [NaN, 100]: the table row may show NaN, but tau ranks 100, the clip FAILs on."""
    subjects = [
        result("R1", "hidden", math.nan, 0.0, InputLevel.ORACLE),
        result("R1", "hidden", 100.0, 0.0, InputLevel.ORACLE, subject="b"),
    ]
    hidden = row(math.nan, scene="s19", clip_id="hidden", subjects=subjects)
    tau = fit_tau([*clean_rows(np.linspace(0.0, 1.0, 19)), hidden], 0.0, ALPHA, 0.0)
    assert (tau.n, tau.n_dropped, tau.value) == (20, 0, 100.0)
    assert fails(hidden, RULE, 1.0)


def test_only_nan_clean_magnitudes_are_unranked_not_absent() -> None:
    nan_rows = [row(math.nan, scene=f"s{i}") for i in range(3)]
    tau = fit_tau(nan_rows, 0.0, ALPHA, 0.0)
    assert (tau.n, tau.n_dropped, tau.value) == (0, 3, None)
    assert check_status(tau, []) == UNRANKED_NEGATIVES


def test_a_scored_row_without_subjects_is_rejected() -> None:
    with pytest.raises(CalibrationDataError, match="no per-subject results"):
        require_subjects([row(0.5, subjects=())])


def test_tau_is_the_conformal_quantile_when_it_exceeds_the_tolerance() -> None:
    values = np.linspace(0.0, 1.8, 19)
    tau = fit_tau(clean_rows(values), 0.5, ALPHA, 1e-5)
    assert tau.value == tau.conformal == pytest.approx(1.8 - 0.5)
    assert fit_tau(clean_rows(values), 0.5, ALPHA, 0.0).value == tau.value


def test_floating_point_noise_falls_back_to_the_tolerance() -> None:
    """Oracle clean ~1e-14: tau = tolerance and neither the cv nor fresh clean scenes fail."""
    rng = np.random.default_rng(2)
    noise = clean_rows(np.abs(rng.normal(0.0, 1e-14, 60)))
    tau = fit_tau(noise, 0.0, ALPHA, 1e-5)
    assert tau.value == 1e-5 and tau.conformal is not None and tau.conformal < 1e-12
    cv = cv_verdicts(noise, RULE, settings(), 1e-5)
    assert all(v is Verdict.PASS for v in cv.values())
    fresh = clean_rows(np.abs(rng.normal(0.0, 1e-14, 200)))
    assert not any(fails(r, RULE, tau.value) for r in fresh)
    without = fit_tau(noise, 0.0, ALPHA, 0.0).value
    assert without is not None and any(fails(r, RULE, without) for r in fresh)


def test_two_clean_clips_of_one_scene_are_rejected() -> None:
    rows = [*clean_rows(np.linspace(0.0, 1.0, 19)), row(0.5, scene="s0", clip_id="dup")]
    with pytest.raises(CalibrationDataError):
        fit_tau(rows, 0.0, ALPHA, 0.0)


def test_cross_validated_flags_cover_each_clean_clip_once() -> None:
    rows = clean_rows(np.random.default_rng(1).normal(0.0, 1.0, 40))
    verdicts = cv_verdicts(rows, RULE, settings(), 0.0)
    assert set(verdicts) == {r.clip_id for r in rows}


def test_the_decided_fpr_excludes_abstentions() -> None:
    verdicts = [Verdict.FAIL, Verdict.PASS, Verdict.ABSTAIN, Verdict.ABSTAIN]
    total, decided = false_positive_rates(verdicts, 0.95)
    assert (total.k, total.n, decided.k, decided.n) == (1, 4, 1, 2)


def test_off_target_failures_flag_an_undeclared_coupling() -> None:
    clean = clean_rows(np.linspace(0.0, 1.0, 20))
    off = [row(5.0, role=Role.OFF_TARGET, scene=f"o{i}", clip_id=f"o{i}") for i in range(10)]
    fpr = fpr_summary([*clean, *off], RULE, 1.0, {}, settings())
    assert fpr.off_target.k == 10 and fpr.undeclared_coupling
    assert fpr.off_target_failures == ("pert",)
    assert fpr.in_sample.n == 20 and fpr.in_sample.k == 0
    assert fpr.in_sample_decided.n == 20


def test_the_off_target_interval_is_clustered_by_scene() -> None:
    off = [
        row(5.0 if s < 3 else 0.0, role=Role.OFF_TARGET, scene=f"o{s}", clip_id=f"o{s}-{c}")
        for s in range(10)
        for c in range(4)
    ]
    fpr = fpr_summary(off, RULE, 1.0, {}, settings())
    rate = fpr.off_target
    assert (rate.k, rate.n, rate.n_scenes, rate.ci_method) == (12, 40, 10, "scene_bootstrap")
    naive = fpr_summary(
        [
            row(5.0 if i < 12 else 0.0, role=Role.OFF_TARGET, scene=f"o{i}", clip_id=f"o{i}")
            for i in range(40)
        ],
        RULE,
        1.0,
        {},
        settings(),
    ).off_target
    assert naive.ci_method == "clopper_pearson"
    assert rate.ci.high - rate.ci.low > naive.ci.high - naive.ci.low  # type: ignore[operator]


def test_tpr_counts_abstentions_as_misses() -> None:
    positives = [
        row(2.0, role=Role.POSITIVE, level=1, mu=1.0, sigma=0.1, clip_id="a"),
        row(1.05, role=Role.POSITIVE, level=1, mu=1.0, sigma=0.1, clip_id="b"),
    ]
    (level,) = tpr_by_level(positives, RULE, 1.0, 0.95)
    assert (level.total.k, level.total.n) == (1, 2)
    assert (level.decided.k, level.decided.n) == (1, 1)


TRUE_LD = 0.1
SLOPE = 4.0
CYCLE = TRUE_LD * np.array([0.5, 1.0, 2.0, 4.0])
DENSE = TRUE_LD * np.array([0.25, 0.5, 1.0, 2.0, 4.0])


def detection_probability(mu: float) -> float:
    """P(d | mu) = expit(b0 + b1 log mu) with P(d | TRUE_LD) = 1 - beta."""
    b0 = math.log(0.95 / 0.05) - SLOPE * math.log(TRUE_LD)
    return 1.0 / (1.0 + math.exp(-(b0 + SLOPE * math.log(mu))))


def cycle_design(rng: np.random.Generator, n_scenes: int) -> list[Detection]:
    """The full design: one level per scene, cycling over the levels."""
    found = []
    for s in range(n_scenes):
        level = s % len(CYCLE)
        mu = float(CYCLE[level])
        found.append(
            Detection(f"s{s}", mu, bool(rng.random() < detection_probability(mu)), level + 1)
        )
    return found


def dense_design(rng: np.random.Generator, n_scenes: int) -> list[Detection]:
    return [
        Detection(f"s{s}", float(mu), bool(rng.random() < detection_probability(mu)), level)
        for s in range(n_scenes)
        for level, mu in enumerate(DENSE, start=1)
    ]


def covers(found: list[Detection]) -> bool:
    """A censored fit counts as a miss; an open side of the interval is unbounded."""
    ld = detection_limit(found, "u", settings())
    if ld.value is None:
        return False
    low = -math.inf if ld.ci.low is None else ld.ci.low
    high = math.inf if ld.ci.high is None else ld.ci.high
    return low <= TRUE_LD <= high


def test_the_detection_limit_interval_covers_in_the_full_design() -> None:
    """AC7 coverage over 60 replicates of 60 scenes, one level per scene: >= 0.85 (the profile
    interval measured 0.93, 0.985, 0.96, 0.97 at 20, 30, 50, 100 scenes over 200 replicates)."""
    rng = np.random.default_rng(0)
    assert float(np.mean([covers(cycle_design(rng, 60)) for _ in range(60)])) >= 0.85


def test_the_detection_limit_point_is_recovered() -> None:
    """AC7 error over 40 replicates of 60 scenes x 5 levels: median relative error <= 15 %."""
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(40):
        ld = detection_limit(dense_design(rng, 60), "u", settings())
        assert ld.censored == "none" and ld.value is not None
        errors.append(abs(ld.value - TRUE_LD) / TRUE_LD)
    assert float(np.median(errors)) <= 0.15


def perfect(n: int) -> list[Detection]:
    return [Detection(f"s{i}", mu, True, lv) for i in range(n) for lv, mu in ((1, 0.05), (2, 0.1))]


@pytest.mark.parametrize("n", [5, 20])
def test_few_perfect_detections_do_not_censor_below(n: int) -> None:
    """5/5 and 20/20 bound the rate at about 0.55 and 0.86 (< 0.95): L_D is not determined."""
    ld = detection_limit(perfect(n), "u", settings())
    assert (ld.censored, ld.value, ld.bound, ld.reason) == ("none", None, None, NOT_DETERMINED)


def test_perfect_detection_with_enough_clips_is_censored_below() -> None:
    """60/60 bounds the one-sided 95 % Clopper-Pearson rate at 0.951 >= 0.95."""
    ld = detection_limit(perfect(60), "u", settings())
    assert (ld.censored, ld.bound, ld.value) == ("below", 0.05, None)


def three_levels(detected_at: frozenset[int]) -> list[Detection]:
    return [
        Detection(f"s{i}", mu, lv in detected_at, lv)
        for i in range(10)
        for lv, mu in ((1, 0.05), (2, 0.1), (3, 0.2))
    ]


def test_nothing_detected_is_censored_above() -> None:
    ld = detection_limit(three_levels(frozenset()), "u", settings())
    assert (ld.censored, ld.bound, ld.value) == ("above", 0.2, None)


def test_a_failed_fit_with_detections_at_the_lowest_level_is_not_determined() -> None:
    """Detected at the lowest level only (decreasing): the fit has no increasing slope, but the
    data do not show L_D above the tested range either."""
    found = three_levels(frozenset({1}))
    found[0] = Detection("s0", 0.05, False, 1)
    ld = detection_limit(found, "u", settings())
    assert (ld.censored, ld.bound, ld.value, ld.reason) == ("none", None, None, NOT_DETERMINED)


def test_no_positive_clip_has_no_detection_limit() -> None:
    ld = detection_limit([], "u", settings())
    assert (ld.censored, ld.reason, ld.n) == ("none", "no_positives", 0)
