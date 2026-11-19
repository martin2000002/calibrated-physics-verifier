"""final_run.sh refuses before any physverify call: a bad stage range, a rehearsal without its own
ledger or writing outside scratch, a real run writing under scratch, a foreign variable."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "h200" / "final_run.sh"


def _run(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items() if not k.startswith("PHYSVERIFY")}
    base.update(
        PHYSVERIFY_H200_ROOT=str(tmp_path),
        PHYSVERIFY_FINAL_CACHE_DIR=str(tmp_path / "cache"),
    )
    base.update(env)
    return subprocess.run(
        ["bash", str(SCRIPT), *args], env=base, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("args", [("11",), ("3", "1"), ("x",), ("9", "11")])
def test_a_bad_stage_range_is_refused(tmp_path: Path, args: tuple[str, ...]) -> None:
    found = _run(tmp_path, *args)
    assert found.returncode == 2 and "usage" in found.stderr


def test_a_rehearsal_needs_a_fitted_calibration(tmp_path: Path) -> None:
    dry = {"PHYSVERIFY_FINAL_DRYRUN": "1", "PHYSVERIFY_FINAL_DRY_CALIBRATION": str(tmp_path)}
    found = _run(tmp_path, **dry)
    assert found.returncode == 2 and "no abstention.json" in found.stderr


def test_a_rehearsal_needs_its_own_ledger_under_scratch(tmp_path: Path) -> None:
    (tmp_path / "abstention.json").write_text("{}")
    dry = {"PHYSVERIFY_FINAL_DRYRUN": "1", "PHYSVERIFY_FINAL_DRY_CALIBRATION": str(tmp_path)}
    found = _run(tmp_path, **dry)
    assert found.returncode == 2 and "its own ledger" in found.stderr
    elsewhere = _run(tmp_path, PHYSVERIFY__SEALED__LEDGER=str(tmp_path / "l.jsonl"), **dry)
    assert elsewhere.returncode == 2 and "writes only under" in elsewhere.stderr


def test_a_real_run_never_writes_under_scratch(tmp_path: Path) -> None:
    found = _run(tmp_path, PHYSVERIFY_FINAL_DIR=str(tmp_path / "scratch" / "f"))
    assert found.returncode == 2 and "only a dry run writes there" in found.stderr


def test_a_foreign_setting_is_refused(tmp_path: Path) -> None:
    found = _run(tmp_path, PHYSVERIFY__KUBRIC_FRESH__VERSION="v2-dev-b3")
    assert found.returncode == 2 and "PHYSVERIFY__KUBRIC_FRESH__VERSION" in found.stderr


def test_a_rehearsal_never_reads_the_counted_render(tmp_path: Path) -> None:
    (tmp_path / "abstention.json").write_text("{}")
    scratch = tmp_path / "scratch"
    found = _run(
        tmp_path,
        PHYSVERIFY_FINAL_DRYRUN="1",
        PHYSVERIFY_FINAL_DRY_CALIBRATION=str(tmp_path),
        PHYSVERIFY__SEALED__LEDGER=str(scratch / "l.jsonl"),
        PHYSVERIFY_FINAL_DIR=str(scratch / "f"),
        PHYSVERIFY_FINAL_DRY_KUBRIC="v3-test-r3b",
    )
    assert found.returncode == 2 and "development data only" in found.stderr
