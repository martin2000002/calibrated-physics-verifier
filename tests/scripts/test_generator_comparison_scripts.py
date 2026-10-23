"""scripts/h200 generator-comparison and AutoEval scripts against fake ``uv`` and ``nvidia-smi``."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from shell_sandbox import Sandbox, executable

from physverify.generator_comparison.status import GATE_LINE, GATE_PASSED

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "h200"
SCRIPT_NAMES = (
    "_common.sh",
    "run_generator_comparison.sh",
    "smoke_autoeval.sh",
    "setup_autoeval.sh",
)
TOOLS = ("bash", "dirname", "grep", "nproc")
PV = "uv run --no-sync physverify --profile precise --device cuda --set runtime.threads=4 "
FAKE_UV = """#!/bin/bash
printf 'uv %s root=%s\\n' "$*" "$PHYSVERIFY__H3__AUTOEVAL__ROOT" >> "$CALLS"
[[ "$*" == *"compare-generators status"* ]] && printf '%s\\n' "${FAKE_STATUS:-}"
[[ -n "${FAIL_ON:-}" && "$*" == *"$FAIL_ON"* ]] && exit 2
exit 0
"""
FAKE_NVIDIA_SMI = """#!/bin/bash
echo "GPU 0: NVIDIA H200 (UUID: GPU-0)"
"""


class AutoevalSandbox(Sandbox):
    """A copied ``scripts/h200`` whose H200 root is the temporary directory."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, SCRIPT_NAMES, TOOLS)
        executable(self.fakes / "uv", FAKE_UV)
        self.env.update({"PHYSVERIFY_H200_ROOT": str(root), "PHYSVERIFY_TIMING_THREADS": "4"})

    def with_gpu(self) -> None:
        executable(self.fakes / "nvidia-smi", FAKE_NVIDIA_SMI)

    def run(self, script: str, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return self.call(script, *args, **env)


@pytest.fixture
def sandbox(tmp_path: Path) -> AutoevalSandbox:
    return AutoevalSandbox(tmp_path)


@pytest.mark.parametrize(
    "name", ["run_generator_comparison.sh", "smoke_autoeval.sh", "setup_autoeval.sh"]
)
def test_scripts_parse_and_are_executable(name: str) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, "-n", str(SCRIPTS / name)], check=True)
    assert os.access(SCRIPTS / name, os.X_OK)


def test_requirements_pin_the_upstream_transformers() -> None:
    lines = (SCRIPTS / "autoeval_requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "transformers==4.28.1" in lines


def test_base_eval_generates_the_base_arm_with_the_autoeval_root(sandbox: AutoevalSandbox) -> None:
    sandbox.with_gpu()
    result = sandbox.run("run_generator_comparison.sh", "base-eval", "--shard", "1/2")
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [
        f"{PV}compare-generators generate --arm base --shard 1/2 root={sandbox.root}"
    ]


def test_an_explicit_autoeval_root_is_kept(sandbox: AutoevalSandbox) -> None:
    result = sandbox.run(
        "run_generator_comparison.sh", "status", PHYSVERIFY__H3__AUTOEVAL__ROOT="/elsewhere"
    )
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [f"{PV}compare-generators status root=/elsewhere"]


EVAL_STEPS = [
    "compare-generators generate --arm ft",
    "compare-generators score --arm base",
    "compare-generators score --arm ft",
    "compare-generators status",
]


def _steps(sandbox: AutoevalSandbox) -> list[str]:
    return [line.removeprefix(PV).split(" root=")[0] for line in sandbox.lines()]


def test_eval_runs_the_comparison_in_order(sandbox: AutoevalSandbox) -> None:
    sandbox.with_gpu()
    result = sandbox.run(
        "run_generator_comparison.sh", "eval", FAKE_STATUS=f"{GATE_LINE}{GATE_PASSED}"
    )
    assert result.returncode == 0, result.stderr
    assert _steps(sandbox) == [
        *EVAL_STEPS,
        "compare-generators judge --split eval",
        "compare-generators compare",
    ]


@pytest.mark.parametrize("gate", ["not run", "failed"])
def test_eval_without_a_passed_gate_still_compares(sandbox: AutoevalSandbox, gate: str) -> None:
    sandbox.with_gpu()
    result = sandbox.run(
        "run_generator_comparison.sh", "eval", FAKE_STATUS=f"campaign c\n{GATE_LINE}{gate}"
    )
    assert result.returncode == 0, result.stderr
    assert "gate has not passed" in result.stderr
    assert _steps(sandbox) == [*EVAL_STEPS, "compare-generators compare"]


@pytest.mark.parametrize("fail_on", ["compare-generators judge", "compare-generators status"])
def test_other_eval_failures_stay_fatal(sandbox: AutoevalSandbox, fail_on: str) -> None:
    sandbox.with_gpu()
    status = f"{GATE_LINE}{GATE_PASSED}"
    result = sandbox.run("run_generator_comparison.sh", "eval", FAKE_STATUS=status, FAIL_ON=fail_on)
    assert result.returncode == 2
    assert "compare-generators compare" not in _steps(sandbox)


@pytest.mark.parametrize("step", ["freeze", "compare", "status"])
def test_cpu_steps_need_no_gpu(sandbox: AutoevalSandbox, step: str) -> None:
    result = sandbox.run("run_generator_comparison.sh", step)
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [f"{PV}compare-generators {step} root={sandbox.root}"]


@pytest.mark.parametrize(
    ("script", "args"),
    [
        ("run_generator_comparison.sh", ("base-eval",)),
        ("run_generator_comparison.sh", ("dev",)),
        ("run_generator_comparison.sh", ("eval",)),
        ("smoke_autoeval.sh", ()),
    ],
)
def test_gpu_steps_stop_without_a_gpu(
    sandbox: AutoevalSandbox, script: str, args: tuple[str, ...]
) -> None:
    result = sandbox.run(script, *args)
    assert result.returncode == 2
    assert "no CUDA GPU" in result.stderr
    assert sandbox.lines() == []


def test_smoke_runs_the_gate(sandbox: AutoevalSandbox) -> None:
    sandbox.with_gpu()
    result = sandbox.run("smoke_autoeval.sh")
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [f"{PV}compare-generators gate-autoeval root={sandbox.root}"]


@pytest.mark.parametrize("args", [(), ("nonsense",), ("eval", "--limit", "2")])
def test_bad_usage_exits_2(sandbox: AutoevalSandbox, args: tuple[str, ...]) -> None:
    sandbox.with_gpu()
    result = sandbox.run("run_generator_comparison.sh", *args)
    assert result.returncode == 2
    assert "usage: run_generator_comparison.sh" in result.stderr
    assert sandbox.lines() == []


def test_setup_autoeval_reads_every_pin_from_the_committed_config() -> None:
    """The sed of setup_autoeval.sh finds each ``[h3.autoeval]`` value it uses in the repo."""
    script = (SCRIPTS / "setup_autoeval.sh").read_text(encoding="utf-8")
    pins = re.search(r'^PINS="(.+)"$', script, re.MULTILINE)
    reader = re.search(r"^autoeval_value\(\) \{\n.*?^\}\n", script, re.MULTILINE | re.DOTALL)
    assert pins is not None and reader is not None
    keys = re.findall(r"autoeval_value (\w+)\)", script)
    repo = SCRIPTS.parents[1]
    with (repo / pins.group(1)).open("rb") as handle:
        expected = tomllib.load(handle)["h3"]["autoeval"]
    program = f'PINS="{pins.group(1)}"\n{reader.group(0)}for key; do autoeval_value "$key"; done'
    result = subprocess.run(
        ["bash", "-c", program, "bash", *keys],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [expected[key] for key in keys]


def test_setup_autoeval_stops_on_a_missing_pin(tmp_path: Path) -> None:
    sandbox = Sandbox(tmp_path, SCRIPT_NAMES, (*TOOLS, "sed"))
    executable(sandbox.fakes / "uv", FAKE_UV)
    executable(sandbox.fakes / "git", '#!/bin/bash\necho "git $*" >> "$CALLS"\nexit 1\n')
    pins = sandbox.repo / "configs" / "studies" / "generator_comparison.toml"
    pins.parent.mkdir(parents=True)
    with (SCRIPTS.parents[1] / "configs" / "studies" / "generator_comparison.toml").open("rb") as f:
        committed = tomllib.load(f)["h3"]["autoeval"]
    values = "".join(f'{k} = "{v}"\n' for k, v in committed.items() if k != "venv")
    pins.write_text(f"[h3.autoeval]\n{values}", encoding="utf-8")
    result = sandbox.call("setup_autoeval.sh", PHYSVERIFY_H200_ROOT=str(tmp_path))
    assert result.returncode == 2
    assert "[h3.autoeval] venv is not set" in result.stderr
    assert sandbox.lines() == []
