"""calibration.json: round trip, versioning, staleness and the threshold projection (AC5, AC11)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from calibration_cases import artifact_of, check_calibration, fitted_artifact, stratum_model

from physverify.calibration.loader import (
    CalibrationError,
    StaleCalibrationError,
    load_calibration,
    parse_calibration,
    projection_name,
    usable_stratum,
)
from physverify.calibration.schema import CALIBRATION_VERSION, CheckCalibration
from physverify.schema.checks import InputLevel
from physverify.verification.runner import (
    CALIBRATED,
    ThresholdTable,
    load_thresholds,
    thresholds_label,
)


@pytest.fixture(scope="module")
def text() -> str:
    return fitted_artifact().to_json()


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(content, encoding="utf-8")
    return path


def test_the_artifact_round_trips_byte_for_byte(text: str) -> None:
    parsed = parse_calibration(text)
    assert parsed.artifact.to_json() == text


def test_the_json_has_no_nan_or_infinity(text: str) -> None:
    """AC5: unknown values are null, never NaN."""
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-finite constant {c}"))


def test_another_major_version_is_refused(text: str) -> None:
    data = json.loads(text)
    data["version"] = "2.0.0"
    with pytest.raises(CalibrationError, match="version"):
        parse_calibration(json.dumps(data))
    data["version"] = "one"
    with pytest.raises(CalibrationError, match="version"):
        parse_calibration(json.dumps(data))


def test_a_newer_minor_version_with_an_extra_field_is_read(text: str) -> None:
    data = json.loads(text)
    major = CALIBRATION_VERSION.split(".")[0]
    data["version"] = f"{major}.99.0"
    data["future_field"] = {"anything": 1}
    data["strata"][0]["checks"]["R1"]["future_metric"] = 0.5
    parsed = parse_calibration(json.dumps(data))
    assert parsed.version == f"{major}.99.0"
    assert parsed.stratum(InputLevel.ORACLE, "fast").tau("R1") is not None


def test_an_unknown_field_at_the_same_version_is_refused(text: str) -> None:
    data = json.loads(text)
    data["future_field"] = 1
    with pytest.raises(CalibrationError, match="invalid"):
        parse_calibration(json.dumps(data))


def test_a_missing_or_foreign_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="calibrate fit"):
        load_calibration(tmp_path / "absent.json")
    with pytest.raises(CalibrationError, match="not a"):
        load_calibration(write(tmp_path, json.dumps({"status": "calibrated"})))


def test_a_stale_artifact_is_refused(tmp_path: Path, text: str) -> None:
    path = write(tmp_path, text)
    assert load_calibration(path, checks_hash="checks").run_id == "run-a"
    with pytest.raises(StaleCalibrationError, match="refit"):
        load_calibration(path, checks_hash="other")


def test_a_missing_stratum_names_the_ones_present(text: str) -> None:
    with pytest.raises(CalibrationError, match="oracle/fast"):
        parse_calibration(text).stratum(InputLevel.PERCEIVED, "fast")


def test_the_projection_is_a_calibrated_threshold_table(tmp_path: Path, text: str) -> None:
    """AC11: only ok checks get a tau; `verify` labels the table calibrated:<source>."""
    calibration = load_calibration(write(tmp_path, text))
    stratum = calibration.stratum(InputLevel.ORACLE, "fast")
    projection = stratum.thresholds_projection()
    table = ThresholdTable.model_validate(projection)
    assert set(table.checks) == {"R1"}
    assert table.artifact is not None and table.artifact.sha256 == calibration.sha256
    out = tmp_path / projection_name(InputLevel.ORACLE, "fast")
    out.write_text(json.dumps(projection), encoding="utf-8")
    loaded = load_thresholds(out, {"R1": 9.0, "R3a": 9.0})
    assert loaded["status"] == CALIBRATED
    assert thresholds_label(loaded).startswith(f"{CALIBRATED}:calibration.json#oracle/fast@")


def test_non_ok_checks_have_no_tau_nor_weight() -> None:
    checks = {
        "ok": check_calibration(tau=0.5, weight=0.6),
        "short": check_calibration(tau=0.5, weight=0.6, status="insufficient"),
    }
    parsed = parse_calibration(artifact_of((stratum_model(checks),)).to_json())
    stratum = parsed.stratum(InputLevel.PERCEIVED, "fast")
    assert (stratum.tau("ok"), stratum.tau("short")) == (0.5, None)
    assert (stratum.weight("ok"), stratum.weight("short")) == (0.6, 0.0)
    assert stratum.entry("short") is None


def test_the_projection_path_is_relative_to_the_calibration_dir(tmp_path: Path, text: str) -> None:
    stratum = load_calibration(write(tmp_path, text)).stratum(InputLevel.ORACLE, "fast")
    assert stratum.thresholds_projection(tmp_path)["artifact"]["path"] == "calibration.json"
    elsewhere = stratum.thresholds_projection(tmp_path / "other")["artifact"]["path"]
    assert Path(elsewhere).is_absolute()


def test_the_stratum_carries_the_checks_hash(text: str) -> None:
    parsed = parse_calibration(text)
    stratum = parsed.stratum(InputLevel.ORACLE, "fast")
    assert stratum.checks_hash == parsed.artifact.checks_hash is not None


def usable(
    path: Path, *, checks_hash: str = "checks", row_step: int = 1, profile: str = "fast"
) -> str | None:
    return usable_stratum(
        path, InputLevel.ORACLE, profile, checks_hash=checks_hash, row_step=row_step
    )


def test_a_stratum_with_one_ok_check_is_usable(tmp_path: Path, text: str) -> None:
    """R3a and R5b are not ok, R1 is: only the stratum used is required (0b.2)."""
    path = write(tmp_path, text)
    assert parse_calibration(text).status == "insufficient"
    assert usable(path) is None


def test_a_stale_or_absent_artifact_is_not_usable(tmp_path: Path, text: str) -> None:
    path = write(tmp_path, text)
    stale = usable(path, checks_hash="other")
    assert stale is not None and "refit" in stale
    absent = usable(tmp_path / "x.json")
    assert absent is not None and "calibrate fit" in absent


def test_another_profile_or_row_step_is_not_usable(tmp_path: Path, text: str) -> None:
    path = write(tmp_path, text)
    other = usable(path, profile="precise")
    assert other is not None and "no stratum" in other
    step = usable(path, row_step=2)
    assert step is not None and "row_step 1" in step


def _artifact_path(
    tmp_path: Path, checks: dict[str, CheckCalibration], threshold: float | None
) -> Path:
    model = stratum_model(checks, InputLevel.ORACLE, score_threshold=threshold)
    return write(tmp_path, artifact_of((model,)).to_json())


def test_no_ok_check_with_weight_is_not_usable(tmp_path: Path) -> None:
    checks = {
        "short": check_calibration(tau=0.5, status="insufficient"),
        "zero": check_calibration(tau=0.5, weight=0.0),
    }
    found = usable(_artifact_path(tmp_path, checks, 0.3))
    assert found is not None and "no check is ok" in found


def test_an_insufficient_aggregate_is_not_usable(tmp_path: Path) -> None:
    checks = {"ok": check_calibration(tau=0.5)}
    found = usable(_artifact_path(tmp_path, checks, None))
    assert found is not None and "insufficient" in found
