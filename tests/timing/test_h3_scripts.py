"""scripts/h200 H3 and AutoEval scripts against fake ``uv`` and ``nvidia-smi``."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from physverify.generator_comparison.status import GATE_LINE, GATE_PASSED

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "h200"
SCRIPT_NAMES = ("_common.sh", "run_h3.sh", "smoke_autoeval.sh", "setup_autoeval.sh")
TOOLS = ("bash", "dirname", "grep", "nproc")
PV = "uv run --no-sync physverify --profile precise --device cuda --set runtime.threads=4 "
FAKE_UV = """#!/bin/bash
printf 'uv %s root=%s\\n' "$*" "$PHYSVERIFY__H3__AUTOEVAL__ROOT" >> "$CALLS"
[[ "$*" == *"h3 status"* ]] && printf '%s\\n' "${FAKE_STATUS:-}"
[[ -n "${FAIL_ON:-}" && "$*" == *"$FAIL_ON"* ]] && exit 2
exit 0
"""
FAKE_NVIDIA_SMI = """#!/bin/bash
echo "GPU 0: NVIDIA H200 (UUID: GPU-0)"
"""


def _executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class Sandbox:
    """A copied ``scripts/h200`` whose H200 root is the temporary directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.calls = root / "calls.log"
        scripts = self.repo / "scripts" / "h200"
        scripts.mkdir(parents=True)
        for name in SCRIPT_NAMES:
            shutil.copy2(SCRIPTS / name, scripts / name)
        tools = root / "tools"
        tools.mkdir()
        for tool in TOOLS:
            found = shutil.which(tool)
            assert found is not None, tool
            (tools / tool).symlink_to(found)
        self.fakes = root / "fakes"
        self.fakes.mkdir()
        _executable(self.fakes / "uv", FAKE_UV)
        self.bash = tools / "bash"
        self.env = {
            "PATH": f"{self.fakes}:{tools}",
            "HOME": str(root),
            "CALLS": str(self.calls),
            "PHYSVERIFY_H200_ROOT": str(root),
            "PHYSVERIFY_TIMING_THREADS": "4",
        }

    def with_gpu(self) -> None:
        _executable(self.fakes / "nvidia-smi", FAKE_NVIDIA_SMI)

    def run(self, script: str, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.bash), str(self.repo / "scripts" / "h200" / script), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def lines(self) -> list[str]:
        if not self.calls.exists():
            return []
        return self.calls.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


@pytest.mark.parametrize("name", ["run_h3.sh", "smoke_autoeval.sh", "setup_autoeval.sh"])
def test_scripts_parse_and_are_executable(name: str) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, "-n", str(SCRIPTS / name)], check=True)
    assert os.access(SCRIPTS / name, os.X_OK)


def test_requirements_pin_the_upstream_transformers() -> None:
    lines = (SCRIPTS / "autoeval_requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "transformers==4.28.1" in lines


def test_base_eval_generates_the_base_arm_with_the_autoeval_root(sandbox: Sandbox) -> None:
    sandbox.with_gpu()
    result = sandbox.run("run_h3.sh", "base-eval", "--shard", "1/2")
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [f"{PV}h3 generate --arm base --shard 1/2 root={sandbox.root}"]


def test_an_explicit_autoeval_root_is_kept(sandbox: Sandbox) -> None:
    result = sandbox.run("run_h3.sh", "status", PHYSVERIFY__H3__AUTOEVAL__ROOT="/elsewhere")
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [f"{PV}h3 status root=/elsewhere"]


EVAL_STEPS = ["h3 generate --arm ft", "h3 score --arm base", "h3 score --arm ft", "h3 status"]


def _steps(sandbox: Sandbox) -> list[str]:
    return [line.removeprefix(PV).split(" root=")[0] for line in sandbox.lines()]


def test_eval_runs_the_comparison_in_order(sandbox: Sandbox) -> None:
    sandbox.with_gpu()
    result = sandbox.run("run_h3.sh", "eval", FAKE_STATUS=f"{GATE_LINE}{GATE_PASSED}")
    assert result.returncode == 0, result.stderr
    assert _steps(sandbox) == [*EVAL_STEPS, "h3 judge --split eval", "h3 compare"]


@pytest.mark.parametrize("gate", ["not run", "failed"])
def test_eval_without_a_passed_gate_still_compares(sandbox: Sandbox, gate: str) -> None:
    sandbox.with_gpu()
    result = sandbox.run("run_h3.sh", "eval", FAKE_STATUS=f"campaign c\n{GATE_LINE}{gate}")
    assert result.returncode == 0, result.stderr
    assert "gate has not passed" in result.stderr
    assert _steps(sandbox) == [*EVAL_STEPS, "h3 compare"]


@pytest.mark.parametrize("fail_on", ["h3 judge", "h3 status"])
def test_other_eval_failures_stay_fatal(sandbox: Sandbox, fail_on: str) -> None:
    sandbox.with_gpu()
    status = f"{GATE_LINE}{GATE_PASSED}"
    result = sandbox.run("run_h3.sh", "eval", FAKE_STATUS=status, FAIL_ON=fail_on)
    assert result.returncode == 2
    assert "h3 compare" not in _steps(sandbox)


@pytest.mark.parametrize("step", ["freeze", "compare", "status"])
def test_cpu_steps_need_no_gpu(sandbox: Sandbox, step: str) -> None:
    result = sandbox.run("run_h3.sh", step)
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [f"{PV}h3 {step} root={sandbox.root}"]


@pytest.mark.parametrize(
    ("script", "args"),
    [
        ("run_h3.sh", ("base-eval",)),
        ("run_h3.sh", ("dev",)),
        ("run_h3.sh", ("eval",)),
        ("smoke_autoeval.sh", ()),
    ],
)
def test_gpu_steps_stop_without_a_gpu(sandbox: Sandbox, script: str, args: tuple[str, ...]) -> None:
    result = sandbox.run(script, *args)
    assert result.returncode == 2
    assert "no CUDA GPU" in result.stderr
    assert sandbox.lines() == []


def test_smoke_runs_the_gate(sandbox: Sandbox) -> None:
    sandbox.with_gpu()
    result = sandbox.run("smoke_autoeval.sh")
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [f"{PV}h3 gate-autoeval root={sandbox.root}"]


@pytest.mark.parametrize("args", [(), ("nonsense",), ("eval", "--limit", "2")])
def test_bad_usage_exits_2(sandbox: Sandbox, args: tuple[str, ...]) -> None:
    sandbox.with_gpu()
    result = sandbox.run("run_h3.sh", *args)
    assert result.returncode == 2
    assert "usage: run_h3.sh" in result.stderr
    assert sandbox.lines() == []
