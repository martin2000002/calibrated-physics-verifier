"""AC10: the threshold policy truth table, the raw policy and where the thresholds come from."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from physverify.core.errors import ConfigError
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.runner import VerificationSettings, build_policy
from physverify.verification.thresholding import ThresholdParams, ThresholdTable, load_thresholds

TAU = 1.0
THRESHOLDS = ThresholdTable.model_validate(
    {"status": "provisional", "source": "test", "checks": {"R1": {"tau": TAU}}}
)
REGISTRY = {"threshold": ThresholdPolicy, "raw": RawPolicy}


def result(
    magnitude: float | None = 0.5,
    sigma: float = 0.1,
    check_id: str = "R1",
    reason: AbstainReason | None = None,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        clip_id="clip",
        subject=("a",),
        frame_span=(0, 3),
        magnitude=magnitude,
        sigma=sigma,
        unit="pen_over_size",
        verdict=Verdict.UNSCORED if reason is None else Verdict.ABSTAIN,
        input_level=InputLevel.ORACLE,
        abstain_reason=reason,
    )


def abstained() -> CheckResult:
    return result(None, math.nan, reason=AbstainReason.OCCLUDED)


@pytest.mark.parametrize(
    ("case", "params", "verdict", "reason", "threshold"),
    [
        (abstained(), ThresholdParams(), Verdict.ABSTAIN, AbstainReason.OCCLUDED, None),
        (result(check_id="R9"), ThresholdParams(), Verdict.UNSCORED, None, None),
        (
            result(1.5, math.nan),
            ThresholdParams(),
            Verdict.ABSTAIN,
            AbstainReason.UNKNOWN_UNCERTAINTY,
            TAU,
        ),
        (result(1.5, math.nan), ThresholdParams(on_unknown_sigma="point"), Verdict.FAIL, None, TAU),
        (result(1.5, 0.1), ThresholdParams(), Verdict.FAIL, None, TAU),
        (result(0.5, 0.1), ThresholdParams(), Verdict.PASS, None, TAU),
        (
            result(0.95, 0.1),
            ThresholdParams(),
            Verdict.ABSTAIN,
            AbstainReason.INTERVAL_CROSSES_THRESHOLD,
            TAU,
        ),
    ],
    ids=["abstain", "no_tau", "nan_abstain", "nan_point", "fail", "pass", "crosses"],
)
def test_threshold_truth_table(
    case: CheckResult,
    params: ThresholdParams,
    verdict: Verdict,
    reason: AbstainReason | None,
    threshold: float | None,
) -> None:
    decided = ThresholdPolicy(params).decide(case, THRESHOLDS)
    assert decided.verdict is verdict
    assert decided.abstain_reason is reason
    assert decided.threshold == threshold
    assert decided.magnitude == case.magnitude or case.magnitude is None


def test_the_interval_touching_tau_from_below_passes() -> None:
    assert ThresholdPolicy().decide(result(TAU, 0.0), THRESHOLDS).verdict is Verdict.PASS


def test_the_raw_policy_leaves_every_result_unscored() -> None:
    policy = RawPolicy()
    assert policy.decide(result(5.0, 0.0), THRESHOLDS) == result(5.0, 0.0)
    assert policy.decide(abstained(), THRESHOLDS) == abstained()


def test_build_policy_validates_its_params() -> None:
    settings = VerificationSettings(policies={"threshold": {"k_sigma": 3.0}})
    policy = build_policy(REGISTRY, settings)
    assert isinstance(policy, ThresholdPolicy)
    assert policy.decide(result(0.5, 0.2), THRESHOLDS).verdict is Verdict.ABSTAIN
    assert isinstance(build_policy(REGISTRY, VerificationSettings(policy="raw")), RawPolicy)
    with pytest.raises(ConfigError):
        build_policy(REGISTRY, VerificationSettings(policies={"threshold": {"k": 1.0}}))
    with pytest.raises(ConfigError):
        build_policy(REGISTRY, VerificationSettings(policy="unknown"))


def test_provisional_thresholds_without_a_calibration_file(tmp_path: Path) -> None:
    thresholds = load_thresholds(tmp_path / "thresholds.json", {"R1": 0.05})
    assert thresholds.status == "provisional"
    assert thresholds.dump()["checks"] == {"R1": {"tau": 0.05}}
    assert thresholds.label == "provisional"


def test_a_calibration_file_takes_precedence(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    table = {"status": "calibrated", "source": "h1-2026", "checks": {"R1": {"tau": 0.2}}}
    path.write_text(json.dumps(table), encoding="utf-8")
    thresholds = load_thresholds(path, {"R1": 0.05})
    assert thresholds.checks["R1"].tau == 0.2
    assert thresholds.label == "calibrated:h1-2026"
    path.write_text(json.dumps({"status": "calibrated"}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_thresholds(path, {})


def calibrated(entry: dict[str, object]) -> ThresholdTable:
    return ThresholdTable.model_validate(
        {"status": "calibrated", "source": "h1", "checks": {"R1": {"tau": TAU, **entry}}}
    )


def test_the_calibrated_bias_is_subtracted_before_the_comparison() -> None:
    policy = ThresholdPolicy()
    assert policy.decide(result(1.2, 0.0), THRESHOLDS).verdict is Verdict.FAIL
    assert policy.decide(result(1.2, 0.0), calibrated({"bias": 0.3})).verdict is Verdict.PASS


def test_the_sigma_factor_widens_the_interval() -> None:
    policy = ThresholdPolicy()
    assert policy.decide(result(1.3, 0.1), THRESHOLDS).verdict is Verdict.FAIL
    widened = policy.decide(result(1.3, 0.1), calibrated({"sigma_factor": 2.0}))
    assert widened.abstain_reason is AbstainReason.INTERVAL_CROSSES_THRESHOLD


def test_an_unmeasured_sigma_factor_abstains() -> None:
    decided = ThresholdPolicy().decide(result(1.5, 0.1), calibrated({"sigma_factor": None}))
    assert (decided.verdict, decided.abstain_reason) == (
        Verdict.ABSTAIN,
        AbstainReason.UNKNOWN_UNCERTAINTY,
    )


def test_a_calibrated_table_may_name_its_artifact(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    table = {**calibrated({}).dump(), "artifact": {"path": "calibration.json", "sha256": "ab"}}
    path.write_text(json.dumps(table), encoding="utf-8")
    assert load_thresholds(path, {}).dump()["artifact"] == {
        "path": "calibration.json",
        "sha256": "ab",
    }
    path.write_text(json.dumps({**table, "artifact": {"path": "x"}}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_thresholds(path, {})


def test_an_invalid_calibration_file_names_its_path(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ConfigError) as raised:
        load_thresholds(path, {})
    reason = "Expecting value: line 1 column 1 (char 0)"
    assert str(raised.value) == f"{path} is not valid JSON: {reason}"
    assert isinstance(raised.value.__cause__, json.JSONDecodeError)
