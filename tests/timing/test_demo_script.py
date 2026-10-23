"""scripts/h200/run_demo.sh against fake ``uv`` and ``nvidia-smi``."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from shell_sandbox import Sandbox, executable

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "h200"
SCRIPT_NAMES = ("_common.sh", "run_demo.sh")
TOOLS = ("bash", "dirname", "grep", "nproc")
PV = "uv run --no-sync physverify --profile precise --device cuda --set runtime.threads=4 "
PV_CPU = PV.replace("--device cuda", "--device cpu")
VIDEO_RUN = ("run", "--prompt", "a ball", "--query", "ball", "--video", "clip.mp4")
FAKE_UV = """#!/bin/bash
printf 'uv %s\\n' "$*" >> "$CALLS"
"""
FAKE_NVIDIA_SMI = """#!/bin/bash
echo "GPU 0: NVIDIA H200 (UUID: GPU-0)"
"""


class DemoSandbox(Sandbox):
    """A copied ``scripts/h200`` whose H200 root is the temporary directory."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, SCRIPT_NAMES, TOOLS)
        executable(self.fakes / "uv", FAKE_UV)
        self.env.update({"PHYSVERIFY_H200_ROOT": str(root), "PHYSVERIFY_TIMING_THREADS": "4"})

    def with_gpu(self) -> None:
        executable(self.fakes / "nvidia-smi", FAKE_NVIDIA_SMI)

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return self.call("run_demo.sh", *args, **env)


@pytest.fixture
def sandbox(tmp_path: Path) -> DemoSandbox:
    return DemoSandbox(tmp_path)


def test_script_parses_and_is_executable() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, "-n", str(SCRIPTS / "run_demo.sh")], check=True)
    assert os.access(SCRIPTS / "run_demo.sh", os.X_OK)


def test_scenes_run_from_the_selection_into_the_h200_root(sandbox: DemoSandbox) -> None:
    sandbox.with_gpu()
    result = sandbox.run("scenes")
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [f"{PV}demo scenes --from-select --out {sandbox.root / 'demo'}"]


def test_a_given_video_is_scored_on_cpu_without_a_gpu(sandbox: DemoSandbox, tmp_path: Path) -> None:
    out = tmp_path / "elsewhere"
    result = sandbox.run(*VIDEO_RUN, PHYSVERIFY_DEMO_OUT=str(out))
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [
        f"{PV_CPU}demo run --out {out} --prompt a ball --query ball --video clip.mp4"
    ]


def test_a_given_video_uses_the_gpu_when_there_is_one(sandbox: DemoSandbox) -> None:
    sandbox.with_gpu()
    result = sandbox.run(*VIDEO_RUN)
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [
        f"{PV}demo run --out {sandbox.root / 'demo'} --prompt a ball --query ball --video clip.mp4"
    ]


@pytest.mark.parametrize("args", [("scenes",), ("run", "--prompt", "a ball", "--query", "ball")])
def test_gpu_modes_stop_without_a_gpu(sandbox: DemoSandbox, args: tuple[str, ...]) -> None:
    result = sandbox.run(*args)
    assert result.returncode == 2
    assert "no CUDA GPU" in result.stderr
    assert sandbox.lines() == []


@pytest.mark.parametrize("args", [(), ("nonsense",)])
def test_bad_usage_exits_2(sandbox: DemoSandbox, args: tuple[str, ...]) -> None:
    result = sandbox.run(*args)
    assert result.returncode == 2
    assert "usage: run_demo.sh" in result.stderr
    assert sandbox.lines() == []
