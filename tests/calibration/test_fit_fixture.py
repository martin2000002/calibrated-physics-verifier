"""End-to-end fit and evaluation on the synthetic raw-policy report (AC1-AC4, AC10, AC12)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from calibration_cases import (
    CHECKS,
    DATASET,
    config,
    dataset_ref,
    fitted_artifact,
    synthetic_report,
)

from physverify.calibration.artifact import CalibrationArtifact
from physverify.calibration.artifact_io import CalibrationError, parse_calibration
from physverify.calibration.collect import level_data
from physverify.calibration.evaluate import (
    SCORE_CHECK,
    EvaluationInputs,
    EvaluationMeta,
    evaluate,
)
from physverify.calibration.fit import LevelData
from physverify.calibration.report import calibration_markdown, curves_csv
from physverify.calibration.thresholds import NOT_DETERMINED
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split

VOLATILE = ("created_utc", "run_id")


@pytest.fixture(scope="module")
def artifact() -> CalibrationArtifact:
    return fitted_artifact()


def stable(artifact: CalibrationArtifact) -> dict[str, object]:
    data = json.loads(artifact.to_json())
    return {k: v for k, v in data.items() if k not in VOLATILE}


def test_the_fit_is_deterministic(artifact: CalibrationArtifact) -> None:
    """AC4: two runs differ only in created_utc and run_id."""
    assert stable(fitted_artifact(run_id="run-b")) == stable(artifact)


def test_the_statuses_follow_the_data(artifact: CalibrationArtifact) -> None:
    """AC2: R1 has clean and perturbed clips, R3a no perturbation, R5b no clean event."""
    (stratum,) = artifact.strata
    statuses = {c: check.status for c, check in stratum.checks.items()}
    assert statuses == {"R1": "ok", "R3a": "no_positives", "R5b": "no_negatives"}
    r1 = stratum.checks["R1"]
    assert r1.tau is not None and r1.weight.value == 1.0
    assert r1.monotonicity.h1_pass
    assert r1.detection_limit.censored == "none" and r1.detection_limit.value is None
    assert r1.detection_limit.reason == NOT_DETERMINED  # 40/40 detected: bound 0.93 < 0.95
    assert stratum.checks["R3a"].tau is not None and stratum.checks["R3a"].weight.value == 0.0
    assert stratum.checks["R5b"].tau is None
    assert stratum.checks["R5b"].abstention_rate.value is None  # no tau: nothing is judged
    assert stratum.aggregate.score_threshold is not None
    assert artifact.status == "insufficient"  # not every check is ok


def test_too_few_clean_scenes_are_insufficient() -> None:
    (stratum,) = fitted_artifact(10).strata
    assert stratum.checks["R1"].status == "insufficient"
    assert stratum.checks["R1"].tau is not None
    assert (
        parse_calibration(fitted_artifact(10).to_json())
        .stratum(InputLevel.ORACLE, "fast")
        .tau("R1")
        is None
    )


def test_the_reports_render(artifact: CalibrationArtifact) -> None:
    markdown = calibration_markdown(artifact)
    assert "R1" in markdown and "estimate" in markdown.lower()
    assert curves_csv(artifact).splitlines()[0].startswith("level")


def held_out_inputs() -> EvaluationInputs:
    calib = synthetic_report(40)
    heldout = synthetic_report(30, seed=7, split=Split.EVAL, prefix="e")
    both = type(calib)(
        calib.results + heldout.results, calib.table + heldout.table, calib.clips + heldout.clips
    )
    real = synthetic_report(20, seed=9, split=Split.EVAL, prefix="r")
    real_data = level_data(InputLevel.ORACLE, real, CHECKS, config(), split=None)
    synthetic = level_data(InputLevel.ORACLE, both, CHECKS, config(), split=None)
    return EvaluationInputs(synthetic, {"real_fixture": real_data})


def meta(profile: str = "fast") -> EvaluationMeta:
    return EvaluationMeta(
        "2026-10-17T00:00:00Z", "eval-a", "calibration.json", dataset_ref(), profile
    )


def test_evaluation_on_held_out_scenes(artifact: CalibrationArtifact) -> None:
    """AC10/AC12: held-out FPR and TPR are reported and the artifact is only read."""
    text = artifact.to_json()
    calibration = parse_calibration(text)
    report = evaluate(calibration, (held_out_inputs(),), meta(), config())
    assert report.artifact_sha256 == calibration.sha256
    (stratum,) = report.strata
    r1 = next(c for c in stratum.checks if c.check_id == "R1")
    assert r1.fpr_clean.n == 30 and r1.fpr_clean.value is not None
    assert r1.fpr_clean.value <= 0.2
    assert [lv.level for lv in r1.tpr_by_level] == [1, 2, 3]
    assert all(lv.total.n == 30 for lv in r1.tpr_by_level)
    assert any(k.check_id == "R1" and k.kendall_tau_b == 1.0 for k in stratum.scene_kendall)
    assert {r.dataset for r in stratum.real} == {"real_fixture"}
    (flags,) = stratum.real_clip_flags
    assert flags.check_id == SCORE_CHECK and flags.fpr.n > 0
    assert artifact.to_json() == text
    assert report.dataset.name == DATASET


def test_evaluation_refuses_another_profile(artifact: CalibrationArtifact) -> None:
    """1.15: clips of another sampling profile are not judged with this artifact."""
    with pytest.raises(CalibrationError, match="profile"):
        evaluate(
            parse_calibration(artifact.to_json()), (held_out_inputs(),), meta("precise"), config()
        )


def test_evaluation_refuses_another_row_step(artifact: CalibrationArtifact) -> None:
    """1.15: the same check as ``open_stratum`` rejects real clips with another time step."""
    item = held_out_inputs()
    real = item.real["real_fixture"]
    slower = LevelData(real.level, tuple(replace(r, row_step=2) for r in real.rows), real.results)
    inputs = EvaluationInputs(item.synthetic, {"real_fixture": slower})
    with pytest.raises(CalibrationError, match="real_fixture clips: .*row_step"):
        evaluate(parse_calibration(artifact.to_json()), (inputs,), meta(), config())
