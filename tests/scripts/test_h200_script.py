"""scripts/h200 run against fake ``uv``, ``docker`` and ``nvidia-smi`` in a copied tree."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from shell_sandbox import Sandbox, executable

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "h200"
SCRIPT_NAMES = ("_common.sh", "setup_cuda.sh", "run_timing.sh")
TOOLS = (
    "bash", "basename", "cat", "dirname", "grep", "head", "ln", "mkdir", "mktemp", "mv", "nproc",
    "rm", "sed", "tee",
)  # fmt: skip
RUN_ID = "20260924T000000-h200"
IMAGE = "kubric:pinned"
H200 = "GPU 0: NVIDIA H200 (UUID: GPU-0)"
PV = "uv run --no-sync physverify --profile precise --device cuda --set runtime.threads="

FAKE_UV = """#!/bin/bash
printf 'uv %s\\n' "$*" >> "$CALLS"
case "$*" in
  *" doctor") echo "threads: 1"; echo "docker_image: $IMAGE" ;;
  *" timing run "*)
    mkdir -p "$RUNS/$RUN_ID/timing"
    echo "run: $RUNS/$RUN_ID"
    exit "${TIMING_EXIT:-0}" ;;
esac
"""
FAKE_DOCKER = """#!/bin/bash
printf 'docker %s\\n' "$*" >> "$CALLS"
case "$1" in
  info) exit "${DOCKER_INFO_EXIT:-0}" ;;
  image) exit "${IMAGE_EXIT:-1}" ;;
esac
"""
FAKE_NVIDIA_SMI = """#!/bin/bash
printf '%s\\n' "$GPU_LIST"
"""


class TimingSandbox(Sandbox):
    """A copy of ``scripts/h200`` in a fake repository, with a PATH of fakes and basic tools."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, SCRIPT_NAMES, TOOLS)
        self.runs = self.repo / "data" / "runs"
        self.env.update(
            {
                "TMPDIR": str(root),
                "RUNS": str(self.runs),
                "RUN_ID": RUN_ID,
                "IMAGE": IMAGE,
                "GPU_LIST": H200,
                "PHYSVERIFY_TIMING_THREADS": "6",
            }
        )

    def install(self, *, nvidia_smi: bool = True) -> None:
        executable(self.fakes / "uv", FAKE_UV)
        executable(self.fakes / "docker", FAKE_DOCKER)
        if nvidia_smi:
            executable(self.fakes / "nvidia-smi", FAKE_NVIDIA_SMI)

    def run(self, script: str = "run_timing.sh", **env: str) -> subprocess.CompletedProcess[str]:
        return self.call(script, **env)

    def uv_calls(self) -> list[str]:
        return [line for line in self.lines() if line.startswith("uv ")]


@pytest.fixture
def sandbox(tmp_path: Path) -> TimingSandbox:
    return TimingSandbox(tmp_path)


def _pv(command: str, threads: int = 6) -> str:
    return f"{PV}{threads} {command}"


SETUP = [
    (
        "uv sync --no-default-groups --group dev --group cu128 --group perception"
        " --group generation --group data"
    ),
    "uv run --no-sync python -c import sys, torch; sys.exit(0 if torch.cuda.is_available() else 2)",
    _pv("extract fetch-weights"),
    "uv run --no-sync physverify --profile fast extract fetch-weights",
    _pv("timing fetch"),
]
RENDER = [
    _pv("synth generate --subset smoke"),
    _pv("synth simulate --subset smoke"),
    _pv("synth perturb --subset smoke"),
]
TIMING_RUN = _pv("timing run --dataset local_video --subset timing_wan21")


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_scripts_parse(name: str) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, "-n", str(SCRIPTS / name)], check=True)


@pytest.mark.parametrize("name", SCRIPT_NAMES[1:])
def test_entry_scripts_are_executable(name: str) -> None:
    assert os.access(SCRIPTS / name, os.X_OK)


@pytest.mark.parametrize("script", ["run_timing.sh", "setup_cuda.sh"])
@pytest.mark.parametrize(
    ("nvidia_smi", "env"),
    [(False, {}), (True, {"GPU_LIST": ""}), (True, {"PHYSVERIFY_TIMING_THREADS": "0"})],
    ids=["no-nvidia-smi", "no-gpu-listed", "bad-threads"],
)
def test_refuses_without_a_usable_gpu_before_uv(
    sandbox: TimingSandbox, script: str, nvidia_smi: bool, env: dict[str, str]
) -> None:
    sandbox.install(nvidia_smi=nvidia_smi)
    result = sandbox.run(script, **env)
    assert result.returncode == 2
    assert ("H200" in result.stderr) or ("PHYSVERIFY_TIMING_THREADS" in result.stderr)
    assert sandbox.uv_calls() == []


def test_full_sequence_with_docker_and_a_missing_image(sandbox: TimingSandbox) -> None:
    sandbox.install()
    result = sandbox.run()
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [
        *SETUP,
        _pv("timing env"),
        "docker info",
        _pv("doctor"),
        f"docker image inspect {IMAGE}",
        _pv("synth pull-image"),
        *RENDER,
        TIMING_RUN,
    ]
    link = sandbox.runs / "h200-timing-latest"
    assert link.is_symlink() and link.readlink() == Path(RUN_ID)
    assert (link / "timing").is_dir()
    assert (sandbox.runs / f"{RUN_ID}.log").read_text(encoding="utf-8").startswith("run: ")
    assert f"run: {sandbox.runs / RUN_ID}" in result.stdout


def test_present_image_is_not_pulled(sandbox: TimingSandbox) -> None:
    sandbox.install()
    assert sandbox.run(IMAGE_EXIT="0").returncode == 0
    assert _pv("synth pull-image") not in sandbox.lines()
    assert sandbox.lines()[-4:] == [*RENDER, TIMING_RUN]


def test_without_docker_only_the_timing_run_follows_setup(sandbox: TimingSandbox) -> None:
    sandbox.install()
    result = sandbox.run(DOCKER_INFO_EXIT="1")
    assert result.returncode == 0
    assert sandbox.lines() == [*SETUP, _pv("timing env"), "docker info", TIMING_RUN]
    assert "kubric_render will be skipped" in result.stderr


def test_threads_default_to_the_available_cpus(sandbox: TimingSandbox) -> None:
    sandbox.install()
    env = dict(sandbox.env)
    del env["PHYSVERIFY_TIMING_THREADS"]
    sandbox.env = env
    assert sandbox.run(DOCKER_INFO_EXIT="1").returncode == 0
    assert sandbox.uv_calls()[-1] == _pv(
        "timing run --dataset local_video --subset timing_wan21", len(os.sched_getaffinity(0))
    )


def test_timing_run_exit_code_is_kept(sandbox: TimingSandbox) -> None:
    sandbox.install()
    result = sandbox.run(DOCKER_INFO_EXIT="1", TIMING_EXIT="1")
    assert result.returncode == 1
    assert (sandbox.runs / "h200-timing-latest").is_symlink()


def test_setup_alone(sandbox: TimingSandbox) -> None:
    sandbox.install()
    assert sandbox.run("setup_cuda.sh").returncode == 0
    assert sandbox.lines() == SETUP
