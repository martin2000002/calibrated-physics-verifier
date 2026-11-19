"""The graded score: a calibrated violation probability per check that orders severity, decided or
not, so candidates of one prompt with the same verdicts still rank."""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

CALIBRATION_TESTS = str(Path(__file__).resolve().parent.parent / "calibration")
if CALIBRATION_TESTS not in sys.path:
    sys.path.insert(0, CALIBRATION_TESTS)

from stratum_cases import check_calibration, stratum_model, write_artifact  # noqa: E402
from wiring import active_checks_hash, checks  # noqa: E402

from physverify.calibration.api import calibration_api  # noqa: E402
from physverify.calibration.artifact import CheckCalibration  # noqa: E402
from physverify.calibration.artifact_io import Stratum, default_artifact_path  # noqa: E402
from physverify.config.models import Settings  # noqa: E402
from physverify.core.errors import UsageError  # noqa: E402
from physverify.finetune.scorers.graded import (  # noqa: E402
    NO_GRADED,
    GradedScorer,
    graded_score,
    monotonicity_weight,
    stratum_weights,
)
from physverify.schema.checks import AbstainReason, CheckResult, Verdict  # noqa: E402
from physverify.stats.intervals import Ci  # noqa: E402

Result = Callable[..., CheckResult]
API = calibration_api()


def _ordering(check: CheckCalibration, low: float | None) -> CheckCalibration:
    mono = check.monotonicity.model_copy(
        update={
            "kendall_tau_b": None if low is None else low + 0.1,
            "ci": Ci(low=low, high=None if low is None else low + 0.2),
        }
    )
    return check.model_copy(update={"monotonicity": mono})


def _stratum(**lows: float | None) -> Stratum:
    calibrated = {
        c: _ordering(check_calibration(tau=1.0, b=2.0, scale=1.0), lo) for c, lo in lows.items()
    }
    return Stratum(model=stratum_model(calibrated), source="test", artifact=None)


def _score(
    stratum: Stratum, results: list[CheckResult], only: frozenset[str] | None = None
) -> float:
    found = graded_score(API, results, stratum, stratum_weights(stratum, monotonicity_weight), only)
    assert found.score is not None
    return found.score


def _crossing(
    make_result: Result, check_id: str, magnitude: float, sigma: float = 0.1
) -> CheckResult:
    result = make_result(check_id, Verdict.ABSTAIN, magnitude, 1.0, sigma)
    return dataclasses.replace(result, abstain_reason=AbstainReason.INTERVAL_CROSSES_THRESHOLD)


def test_the_weight_is_the_lower_bound_of_the_within_scene_ordering_normalised() -> None:
    weights = stratum_weights(_stratum(R1=0.8, R2=0.4, R3a=-0.2, R5a=None), monotonicity_weight)
    assert weights == {"R1": 1.0, "R2": 0.5, "R3a": 0.0, "R5a": 0.0}


def test_the_score_falls_strictly_with_the_magnitude(make_result: Result) -> None:
    stratum = _stratum(R1=0.5)
    scores = [
        _score(stratum, [make_result("R1", Verdict.PASS, m, 1.0, 0.1)]) for m in (0.2, 0.6, 1.4)
    ]
    assert scores[0] > scores[1] > scores[2]


def test_an_interval_that_crosses_tau_still_grades_and_ranks(make_result: Result) -> None:
    stratum = _stratum(R1=0.5)
    low = graded_score(API, [_crossing(make_result, "R1", 0.9)], stratum, {"R1": 1.0})
    high = graded_score(API, [_crossing(make_result, "R1", 1.1)], stratum, {"R1": 1.0})
    assert low.decided == high.decided == frozenset({"R1"})
    assert low.score is not None and high.score is not None and low.score > high.score
    assert (low.n_failed, high.n_failed) == (0, 1)


def test_a_structural_abstention_does_not_grade(make_result: Result) -> None:
    stratum = _stratum(R1=0.5)
    missing = make_result("R1", Verdict.ABSTAIN, 0.3, 1.0, 0.1)
    found = graded_score(API, [missing], stratum, {"R1": 1.0})
    assert (found.score, found.abstained, found.reason) == (None, True, NO_GRADED)
    assert found.coverage == 0.0


def test_a_check_that_does_not_order_severity_has_no_vote(make_result: Result) -> None:
    stratum = _stratum(R1=0.5, R2=-0.1)
    weights = stratum_weights(stratum, monotonicity_weight)
    found = graded_score(
        API,
        [
            make_result("R1", Verdict.PASS, 0.2, 1.0, 0.1),
            make_result("R2", Verdict.FAIL, 5.0, 1.0, 0.1),
        ],
        stratum,
        weights,
    )
    assert found.decided == frozenset({"R1"}) and found.coverage == 1.0


def test_only_restricts_to_the_common_checks(make_result: Result) -> None:
    stratum = _stratum(R1=0.5, R2=0.5)
    results = [
        make_result("R1", Verdict.PASS, 0.2, 1.0, 0.1),
        make_result("R2", Verdict.FAIL, 2.0, 1.0, 0.1),
    ]
    alone = _score(stratum, results[:1])
    assert _score(stratum, results, only=frozenset({"R1"})) == pytest.approx(alone)
    assert _score(stratum, results) < alone


def test_the_largest_subject_grades(make_result: Result) -> None:
    stratum = _stratum(R1=0.5)
    pair = [
        make_result("R1", Verdict.PASS, 0.2, 1.0, 0.1),
        make_result("R1", Verdict.PASS, 0.7, 1.0, 0.1),
    ]
    assert _score(stratum, pair) == pytest.approx(_score(stratum, pair[1:]))


def test_sigma_is_the_delta_method_of_the_score(make_result: Result) -> None:
    stratum = _stratum(R1=0.5)
    weights = {"R1": 1.0}
    m, s, h = 0.8, 0.1, 1e-6
    found = graded_score(API, [make_result("R1", Verdict.PASS, m, 1.0, s)], stratum, weights)
    up = graded_score(API, [make_result("R1", Verdict.PASS, m + h, 1.0, s)], stratum, weights).score
    down = graded_score(
        API, [make_result("R1", Verdict.PASS, m - h, 1.0, s)], stratum, weights
    ).score
    assert up is not None and down is not None and found.sigma is not None
    assert found.sigma == pytest.approx(abs(up - down) / (2 * h) * s, rel=1e-5)


def test_an_unmeasured_error_does_not_grade(make_result: Result) -> None:
    check = _ordering(check_calibration(tau=1.0, factor=None), 0.5)
    stratum = Stratum(model=stratum_model({"R1": check}), source="test", artifact=None)
    found = graded_score(
        API, [make_result("R1", Verdict.PASS, 0.2, 1.0, 0.1)], stratum, {"R1": 1.0}
    )
    assert (found.abstained, found.reason) == (True, NO_GRADED)


def test_the_graded_scorer_opens_the_calibrated_gate(
    settings: Settings, make_result: Result
) -> None:
    fields = {"profile": settings.profile, "checks_hash": active_checks_hash(settings)}
    write_artifact(default_artifact_path(settings), **fields)
    scorer = GradedScorer(settings, checks())
    assert scorer.unavailable() is None
    assert scorer.thresholds().startswith("calibrated:")
    found = scorer.score([make_result("R1", Verdict.PASS, 0.1, 1.0, 0.01)])
    assert found.reason == NO_GRADED
    assert (GradedScorer.name, GradedScorer.version, GradedScorer.calibrated) == (
        "graded",
        "1",
        True,
    )


def test_no_check_that_orders_severity_abstains(make_result: Result) -> None:
    stratum = _stratum(R1=-0.1, R2=None)
    weights = stratum_weights(stratum, monotonicity_weight)
    assert set(weights.values()) == {0.0}
    found = graded_score(API, [make_result("R1", Verdict.FAIL, 5.0, 1.0, 0.1)], stratum, weights)
    assert (found.score, found.abstained, found.reason) == (None, True, NO_GRADED)


def test_the_graded_scorer_refuses_without_a_calibration(
    settings: Settings, make_result: Result
) -> None:
    scorer = GradedScorer(settings, checks())
    assert scorer.unavailable() is not None
    with pytest.raises(UsageError):
        scorer.score([make_result("R1", Verdict.PASS, 0.1, 1.0, 0.01)])
