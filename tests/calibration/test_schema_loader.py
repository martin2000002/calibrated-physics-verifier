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
)
from physverify.calibration.schema import CALIBRATION_VERSION
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
