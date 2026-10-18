"""scripts/h200/run_demo.sh against fake ``uv`` and ``nvidia-smi``."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

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

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.bash), str(self.repo / "scripts" / "h200" / "run_demo.sh"), *args],
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


def test_script_parses_and_is_executable() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, "-n", str(SCRIPTS / "run_demo.sh")], check=True)
    assert os.access(SCRIPTS / "run_demo.sh", os.X_OK)


def test_scenes_run_from_the_selection_into_the_h200_root(sandbox: Sandbox) -> None:
    sandbox.with_gpu()
    result = sandbox.run("scenes")
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [f"{PV}demo scenes --from-select --out {sandbox.root / 'demo'}"]


def test_a_given_video_is_scored_on_cpu_without_a_gpu(sandbox: Sandbox, tmp_path: Path) -> None:
    out = tmp_path / "elsewhere"
    result = sandbox.run(*VIDEO_RUN, PHYSVERIFY_DEMO_OUT=str(out))
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [
        f"{PV_CPU}demo run --out {out} --prompt a ball --query ball --video clip.mp4"
    ]


def test_a_given_video_uses_the_gpu_when_there_is_one(sandbox: Sandbox) -> None:
    sandbox.with_gpu()
    result = sandbox.run(*VIDEO_RUN)
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [
        f"{PV}demo run --out {sandbox.root / 'demo'} --prompt a ball --query ball --video clip.mp4"
    ]


@pytest.mark.parametrize("args", [("scenes",), ("run", "--prompt", "a ball", "--query", "ball")])
def test_gpu_modes_stop_without_a_gpu(sandbox: Sandbox, args: tuple[str, ...]) -> None:
    result = sandbox.run(*args)
    assert result.returncode == 2
    assert "no CUDA GPU" in result.stderr
    assert sandbox.lines() == []


@pytest.mark.parametrize("args", [(), ("nonsense",)])
def test_bad_usage_exits_2(sandbox: Sandbox, args: tuple[str, ...]) -> None:
    result = sandbox.run(*args)
    assert result.returncode == 2
    assert "usage: run_demo.sh" in result.stderr
    assert sandbox.lines() == []
