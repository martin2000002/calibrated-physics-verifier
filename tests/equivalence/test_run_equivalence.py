"""`tools/equivalence/run_equivalence.sh` fails when a side's setup fails instead of passing."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "equivalence" / "run_equivalence.sh"
# Stands in for the checkout's python running driver.py: lists the scenarios of a file next to
# it, normalizes to the same single file on both sides, and fails `prepare` on request.
FAKE_PYTHON = """#!/usr/bin/env bash
here="$(dirname "$0")"
shift
case "$1" in
    list) cat "$here/scenarios" ;;
    prepare) [[ ! -e "$here/fail_prepare" ]] ;;
    normalize) mkdir -p "$3" && echo same >"$3/file" ;;
esac
"""


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    (repo / ".venv" / "bin").mkdir(parents=True)
    python = repo / ".venv" / "bin" / "python"
    python.write_text(FAKE_PYTHON)
    python.chmod(0o755)
    (repo / "configs").mkdir()
    (repo / "configs" / "default.toml").write_text("")
    _git(repo, "init", "--quiet")
    _git(repo, "add", "configs")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-m", "init")
    return repo


def _run(repo: Path, work: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "EQ_REPO": str(repo), "EQ_WORKDIR": str(work), "EQ_EXPECTED": ""}
    return subprocess.run(
        ["bash", str(SCRIPT), "HEAD", "HEAD"], env=env, capture_output=True, text=True, check=False
    )


def test_identical_sides_pass(fake_repo: Path, tmp_path: Path) -> None:
    (fake_repo / ".venv" / "bin" / "scenarios").write_text("one\n")
    result = _run(fake_repo, tmp_path / "work")
    assert result.returncode == 0, result.stderr
    assert "normalized outputs identical" in result.stdout


def test_an_empty_scenario_list_fails(fake_repo: Path, tmp_path: Path) -> None:
    (fake_repo / ".venv" / "bin" / "scenarios").write_text("")
    result = _run(fake_repo, tmp_path / "work")
    assert result.returncode == 1
    assert "no scenarios listed" in result.stderr


def test_a_failing_setup_step_fails(fake_repo: Path, tmp_path: Path) -> None:
    (fake_repo / ".venv" / "bin" / "scenarios").write_text("one\n")
    (fake_repo / ".venv" / "bin" / "fail_prepare").write_text("")
    assert _run(fake_repo, tmp_path / "work").returncode == 1


def test_an_unknown_revision_fails(fake_repo: Path, tmp_path: Path) -> None:
    (fake_repo / ".venv" / "bin" / "scenarios").write_text("one\n")
    env = {**os.environ, "EQ_REPO": str(fake_repo), "EQ_WORKDIR": str(tmp_path / "w")}
    result = subprocess.run(
        ["bash", str(SCRIPT), "no-such-rev", "HEAD"],
        env={**env, "EQ_EXPECTED": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
