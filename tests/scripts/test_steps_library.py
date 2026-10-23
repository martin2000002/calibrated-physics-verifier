"""scripts/h200/_steps.sh sourced on its own: step rows, tolerated codes and resume checks."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

STEPS_SH = Path(__file__).resolve().parents[2] / "scripts" / "h200" / "_steps.sh"


def run_steps(tmp_path: Path, body: str, steps: str = "") -> subprocess.CompletedProcess[str]:
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "notes.md").touch()
    (tmp_path / "steps.tsv").write_text(steps, encoding="utf-8")
    script = "\n".join(
        [
            "set -euo pipefail",
            "die() { printf '%s\\n' \"$*\" >&2; exit 2; }",
            f'source "{STEPS_SH}"',
            f'LOGS="{tmp_path / "logs"}" STATE="{tmp_path / "state.env"}"',
            f'STEPS="{tmp_path / "steps.tsv"}" NOTES="{tmp_path / "notes.md"}"',
            f'SUMMARY="{tmp_path / "summary.md"}" SUMMARY_TITLE=t SUMMARY_INFO=i',
            body,
        ]
    )
    return subprocess.run(
        ["bash", "-c", script],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )


def rows(tmp_path: Path) -> list[list[str]]:
    lines = (tmp_path / "steps.tsv").read_text(encoding="utf-8").splitlines()
    return [line.split("\t") for line in lines]


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ("", "step 2 has not completed (last exit 'none')"),
        ("2\tstages\t0\t1\t-\n2\tstages\t2\t1\t-\n", "step 2 has not completed (last exit '2')"),
        ("2\tstages\t1\t1\t-\n", "step 2 has not completed (last exit '1')"),
        (
            "2\tstages\t0\t1\t-\n2\tstages\trunning\t0\t-\n",
            "step 2 has not completed (last exit 'running')",
        ),
    ],
)
def test_a_start_after_an_incomplete_step_is_refused(
    tmp_path: Path, steps: str, message: str
) -> None:
    result = run_steps(tmp_path, "require_completed 2 3", steps)
    assert result.returncode == 2
    assert message in result.stderr and "rerun from 2" in result.stderr


def test_a_step_rerun_successfully_counts_as_completed(tmp_path: Path) -> None:
    result = run_steps(
        tmp_path, "require_completed 0 1", "0\tmigrate\t2\t1\t-\n0\tmigrate\t0\t1\t-\n"
    )
    assert result.returncode == 0, result.stderr


def test_a_step_without_its_saved_state_is_refused(tmp_path: Path) -> None:
    done = "2\tstages\t0\t1\t-\n"
    body = "STEP_SAVES=([2]=STAGES_RUN)\nrequire_completed 2 3"
    result = run_steps(tmp_path, body, done)
    assert result.returncode == 2
    assert "step 2 recorded no STAGES_RUN" in result.stderr
    saved = run_steps(tmp_path, "STEP_SAVES=([2]=STAGES_RUN)\nsave STAGES_RUN r1\n" + body, done)
    assert saved.returncode == 0, saved.stderr


def test_a_tolerated_code_is_noted_and_the_step_goes_on(tmp_path: Path) -> None:
    body = "begin 1 extract\nrun_logged 1 -- bash -c 'exit 1'\nrow 0"
    result = run_steps(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert [row[:3] for row in rows(tmp_path)] == [
        ["1", "extract", "running"],
        ["1", "extract", "0"],
    ]
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "step 1 (extract): some clips failed (exit 1)" in summary


@pytest.mark.parametrize(("tolerated", "code"), [("0", "1"), ("1", "2"), ("75", "75")])
def test_a_fatal_code_stops_the_step_with_it(tmp_path: Path, tolerated: str, code: str) -> None:
    body = f"begin 5 fit\nrun_logged {tolerated} -- bash -c 'exit {code}'\nrow 0"
    result = run_steps(tmp_path, body)
    assert result.returncode == int(code)
    assert rows(tmp_path)[-1][:3] == ["5", "fit", code]
    assert "step 5 (fit) failed" in result.stderr
