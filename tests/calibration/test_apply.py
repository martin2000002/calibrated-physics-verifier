"""Calibrated verdicts, p_c and the clip score S with its delta-method sigma."""

from __future__ import annotations

import math
from dataclasses import fields

import pytest
from calibration_cases import check_calibration, result, view

from physverify.calibration.apply import (
    NO_DECIDED,
    NO_INFORMATIVE,
    ClipScore,
    calibrated_result,
    score_clip,
    violation_probability,
)
from physverify.calibration.artifact_io import KSigmaMismatchError
from physverify.schema.checks import AbstainReason, CheckResult, Verdict

K_SIGMA = 2.0
CHECKS = {
    "A": check_calibration(tau=0.5, a=-1.0, b=3.0, scale=0.2, weight=1.0, factor=1.5),
    "B": check_calibration(tau=1.0, a=0.5, b=2.0, scale=0.5, weight=0.4, bias=0.1, factor=2.0),
    "C": check_calibration(tau=0.2, a=-2.0, b=5.0, scale=0.1, weight=0.7),
}
MAGNITUDES = {"A": 0.3, "B": 1.6, "C": 0.05}
SIGMAS = {"A": 0.02, "B": 0.05, "C": 0.01}


def results(
    magnitudes: dict[str, float] = MAGNITUDES, sigmas: dict[str, float] = SIGMAS
) -> list[CheckResult]:
    return [result(c, "clip", m, sigmas[c]) for c, m in magnitudes.items()]


def test_sigma_s_matches_central_finite_differences() -> None:
    """AC17: sigma_S from the delta method within 1 % of sum_c (dS/dm_c factor_c sigma_c)^2."""
    stratum = view(CHECKS)
    found = score_clip(results(), stratum, K_SIGMA)
    assert found.score is not None and found.sigma is not None
    assert set(found.decided_checks) == set(CHECKS) and found.n_failed == 1
    total = 0.0
    for check_id, m in MAGNITUDES.items():
        h = 1e-6
        up = score_clip(results({**MAGNITUDES, check_id: m + h}), stratum, K_SIGMA).score
        down = score_clip(results({**MAGNITUDES, check_id: m - h}), stratum, K_SIGMA).score
        assert up is not None and down is not None
        factor = CHECKS[check_id].sigma.factor
        assert factor is not None
        total += ((up - down) / (2 * h) * factor * SIGMAS[check_id]) ** 2
    assert found.sigma == pytest.approx(math.sqrt(total), rel=0.01)


def test_s_is_the_weighted_product_of_one_minus_p() -> None:
    stratum = view(CHECKS)
    found = score_clip(results(), stratum, K_SIGMA)
    expected = 1.0
    for check_id, m in MAGNITUDES.items():
        p = violation_probability(CHECKS[check_id], m)
        assert p is not None and found.per_check[check_id] == pytest.approx(p)
        expected *= (1.0 - p) ** CHECKS[check_id].weight.value
    assert found.score == pytest.approx(expected)
    assert (found.coverage, found.abstained, found.reason) == (1.0, False, None)


def test_s_decreases_as_a_magnitude_grows() -> None:
    stratum = view(CHECKS)
    scores = [
        score_clip(results({**MAGNITUDES, "B": m}), stratum, K_SIGMA).score
        for m in (1.3, 1.5, 1.8, 2.5)
    ]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


def test_the_oracle_sigma_is_zero() -> None:
    found = score_clip(results(sigmas=dict.fromkeys(SIGMAS, 0.0)), view(CHECKS), K_SIGMA)
    assert found.sigma == 0.0


def test_no_decided_check_abstains_with_zero_coverage() -> None:
    crossing = {"A": 0.5, "B": 1.1, "C": 0.2}
    found = score_clip(results(crossing), view(CHECKS), K_SIGMA)
    assert (found.score, found.sigma, found.coverage) == (None, None, 0.0)
    assert found.abstained and found.reason == NO_DECIDED
    assert set(found.abstained_checks) == set(CHECKS)


def test_only_restricts_the_checks() -> None:
    found = score_clip(results(), view(CHECKS), K_SIGMA, only={"A", "C"})
    assert found.decided_checks == ("A", "C")
    assert set(found.per_check) == {"A", "C"}


def test_a_weight_zero_check_does_not_move_s() -> None:
    zero = {**CHECKS, "B": check_calibration(tau=1.0, a=0.5, b=2.0, scale=0.5, weight=0.0)}
    with_b = score_clip(results(), view(zero), K_SIGMA).score
    without_b = score_clip(results(), view(zero), K_SIGMA, only={"A", "C"}).score
    assert with_b == pytest.approx(without_b)


def test_no_informative_check_has_no_score() -> None:
    flat = {c: check_calibration(tau=0.5, weight=0.0) for c in CHECKS}
    found = score_clip(results(), view(flat), K_SIGMA)
    assert (found.score, found.reason, found.abstained) == (None, NO_INFORMATIVE, True)


def test_a_check_without_calibration_stays_unscored() -> None:
    stratum = view({"A": CHECKS["A"], "B": check_calibration(tau=None, status="insufficient")})
    assert calibrated_result(result("B", magnitude=5.0), stratum, K_SIGMA).verdict is (
        Verdict.UNSCORED
    )
    found = score_clip(results(), stratum, K_SIGMA)
    assert found.decided_checks == ("A",)


def test_an_unmeasured_error_abstains() -> None:
    stratum = view({"A": check_calibration(tau=0.5, factor=None)})
    found = calibrated_result(result("A", magnitude=0.9, sigma=0.01), stratum, K_SIGMA)
    assert (found.verdict, found.abstain_reason) == (
        Verdict.ABSTAIN,
        AbstainReason.UNKNOWN_UNCERTAINTY,
    )
    assert math.isnan(found.sigma)


def test_the_a11_adapter_maps_the_clip_score_field_by_field() -> None:
    calibrated = pytest.importorskip("physverify.finetune.scorers.calibrated")
    found = score_clip(results(), view(CHECKS), K_SIGMA)
    aggregate = calibrated.to_aggregate(found)
    assert (aggregate.score, aggregate.sigma, aggregate.coverage) == (
        found.score,
        found.sigma,
        found.coverage,
    )
    assert aggregate.decided == frozenset(found.decided_checks)
    assert (aggregate.n_decided, aggregate.n_failed) == (3, found.n_failed)
    assert aggregate.per_check == dict(found.per_check)
    assert {f.name for f in fields(ClipScore)} == {
        "clip_id",
        "score",
        "sigma",
        "coverage",
        "abstained",
        "reason",
        "n_failed",
        "decided_checks",
        "abstained_checks",
        "per_check",
    }


def test_k_sigma_defaults_to_the_one_the_stratum_was_fitted_with() -> None:
    stratum = view(CHECKS)
    assert stratum.model.k_sigma == K_SIGMA
    assert score_clip(results(), stratum) == score_clip(results(), stratum, K_SIGMA)


def test_another_k_sigma_is_refused() -> None:
    with pytest.raises(KSigmaMismatchError, match="k_sigma"):
        score_clip(results(), view(CHECKS), K_SIGMA + 1.0)
    with pytest.raises(KSigmaMismatchError, match="k_sigma"):
        calibrated_result(result("A", magnitude=0.3), view(CHECKS), 1.0)
