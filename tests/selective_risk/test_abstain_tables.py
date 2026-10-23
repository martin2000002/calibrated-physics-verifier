"""Abstain reads one calibration stratum per input level."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CALIBRATION_TESTS = str(Path(__file__).resolve().parent.parent / "calibration")
if CALIBRATION_TESTS not in sys.path:
    sys.path.insert(0, CALIBRATION_TESTS)

from calibration_cases import PROFILE, artifact_of, check_calibration, stratum_model  # noqa: E402

from physverify.core.errors import MissingInputError, UsageError  # noqa: E402
from physverify.schema.checks import InputLevel  # noqa: E402
from physverify.selective_risk.curve import require_taus  # noqa: E402
from physverify.selective_risk.tables import level_tables  # noqa: E402

FALLBACK = {"status": "provisional", "source": "configs", "checks": {"R1": {"tau": 0.5}}}


def write_artifact(tmp_path: Path) -> Path:
    oracle = stratum_model({"R1": check_calibration(tau=0.1)}, level=InputLevel.ORACLE)
    perceived = stratum_model(
        {"R1": check_calibration(tau=0.3, bias=0.02, factor=1.5)}, level=InputLevel.PERCEIVED
    )
    path = tmp_path / "calibration.json"
    path.write_text(artifact_of((oracle, perceived)).to_json(), encoding="utf-8")
    return path


def test_each_level_gets_its_own_stratum(tmp_path: Path) -> None:
    tables = level_tables(write_artifact(tmp_path), PROFILE, FALLBACK, {})
    oracle, perceived = tables[InputLevel.ORACLE], tables[InputLevel.PERCEIVED]
    assert oracle["checks"]["R1"]["tau"] == pytest.approx(0.1)
    assert perceived["checks"]["R1"]["tau"] == pytest.approx(0.3)
    assert perceived["checks"]["R1"]["sigma_factor"] == pytest.approx(1.5)
    assert oracle["status"] == perceived["status"] == "calibrated"
    assert perceived["artifact"]["path"] == "calibration.json"


def test_without_an_artifact_both_levels_use_the_fallback(tmp_path: Path) -> None:
    for artifact in (None, tmp_path / "absent.json"):
        tables = level_tables(artifact, PROFILE, FALLBACK, {})
        assert tables[InputLevel.ORACLE] == tables[InputLevel.PERCEIVED] == FALLBACK


def test_an_explicit_table_overrides_one_level(tmp_path: Path) -> None:
    table = tmp_path / "oracle.json"
    table.write_text(
        '{"status": "calibrated", "source": "mine", "checks": {"R1": {"tau": 0.7}}}',
        encoding="utf-8",
    )
    tables = level_tables(write_artifact(tmp_path), PROFILE, FALLBACK, {InputLevel.ORACLE: table})
    assert tables[InputLevel.ORACLE]["checks"]["R1"]["tau"] == pytest.approx(0.7)
    assert tables[InputLevel.PERCEIVED]["checks"]["R1"]["tau"] == pytest.approx(0.3)
    with pytest.raises(UsageError, match="no oracle threshold table"):
        level_tables(None, PROFILE, FALLBACK, {InputLevel.ORACLE: tmp_path / "missing.json"})


def test_strata_without_an_ok_check_are_refused_with_a_clear_error(tmp_path: Path) -> None:
    """The first calibration fit: every check insufficient, so both projections have no checks."""
    insufficient = {"R1": check_calibration(tau=None, status="insufficient")}
    strata = tuple(
        stratum_model(insufficient, level=level)
        for level in (InputLevel.ORACLE, InputLevel.PERCEIVED)
    )
    path = tmp_path / "calibration.json"
    path.write_text(artifact_of(strata).to_json(), encoding="utf-8")
    tables = level_tables(path, PROFILE, FALLBACK, {})
    assert tables[InputLevel.ORACLE]["checks"] == {}
    with pytest.raises(
        MissingInputError, match=r"oracle threshold table .* has no tau for any check"
    ):
        require_taus(tables[InputLevel.ORACLE], InputLevel.ORACLE)
    require_taus(FALLBACK, InputLevel.PERCEIVED)
