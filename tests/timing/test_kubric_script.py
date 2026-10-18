"""scripts/h200/run_kubric.sh against fake ``udocker``, ``uv`` and ``df`` in a copied tree."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "h200"
SCRIPT_NAMES = ("_common.sh", "run_kubric.sh")
TOOLS = (
    "bash", "basename", "cat", "date", "dirname", "env", "grep", "ln", "mkdir", "nproc", "tail",
    "sort", "tee", "tr",
)  # fmt: skip
PV = (
    "uv run --no-sync physverify --profile precise --set runtime.threads=16 "
    "--set dataset.render.workers=12 --set dataset.render.jobs_per_process=8 "
)
DD_VARIABLES = (
    "DD_TRACE_ENABLED",
    "DD_SYMBOL_DATABASE_UPLOAD_ENABLED",
    "DD_INSTRUMENTATION_TELEMETRY_ENABLED",
    "DD_REMOTE_CONFIGURATION_ENABLED",
)

FAKE_UV = """#!/bin/bash
printf 'uv %s\\n' "$*" >> "$CALLS"
env | grep -E '^(PHYSVERIFY__|DD_)' | sort > "$ENV_DUMP"
case "$*" in
  *" doctor") echo "docker_image: x"; echo "container_engine: ${DOCTOR_ENGINE:-udocker (kub)}" ;;
  *" dataset build "*) echo "render: 0 launched"; exit "${BUILD_EXIT:-0}" ;;
  *" dataset list "*) echo "render 2 done" ;;
esac
"""
FAKE_UDOCKER = """#!/bin/bash
printf 'udocker %s\\n' "$*" >> "$CALLS"
case "$1" in
  inspect) exit "${INSPECT_EXIT:-0}" ;;
  run) exit "${IMPORT_EXIT:-0}" ;;
esac
"""
FAKE_DF = """#!/bin/bash
printf ' Avail\\n %s\\n' "${FREE_GB:-500}"
"""


def _executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.repo = root / "repo"
        self.data = root / "kubric-data"
        self.calls = root / "calls.log"
        self.env_dump = root / "env.txt"
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
        fakes = root / "fakes"
        fakes.mkdir()
        self.udocker = fakes / "udocker"
        _executable(fakes / "uv", FAKE_UV)
        _executable(self.udocker, FAKE_UDOCKER)
        _executable(fakes / "df", FAKE_DF)
        self.bash = tools / "bash"
        self.env = {
            "PATH": f"{fakes}:{tools}",
            "HOME": str(root),
            "CALLS": str(self.calls),
            "ENV_DUMP": str(self.env_dump),
            "PHYSVERIFY_H200_ENV": str(root / "no-env.sh"),
            "PHYSVERIFY_UDOCKER": str(self.udocker),
            "PHYSVERIFY_KUBRIC_LOCK": str(root / "kubric.lock"),
            "PHYSVERIFY__PATHS__DATA_DIR": str(self.data),
        }

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.bash), str(self.repo / "scripts" / "h200" / "run_kubric.sh"), *args],
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

    def exported(self) -> dict[str, str]:
        pairs = (line.split("=", 1) for line in self.env_dump.read_text().splitlines())
        return dict(pairs)


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def test_script_parses_and_is_executable() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, "-n", str(SCRIPTS / "run_kubric.sh")], check=True)
    assert os.access(SCRIPTS / "run_kubric.sh", os.X_OK)


def test_full_run(sandbox: Sandbox) -> None:
    result = sandbox.run()
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [
        "udocker inspect kub",
        "udocker run --nobanner kub python3 -c import kubric, bpy, pybullet",
        f"{PV}doctor",
        f"{PV}dataset build --subset full",
        f"{PV}dataset list --subset full",
    ]
    exported = sandbox.exported()
    assert exported["PHYSVERIFY__CONTAINER__ENGINE"] == "udocker"
    assert exported["PHYSVERIFY__CONTAINER__UDOCKER_BIN"] == str(sandbox.udocker)
    assert exported["PHYSVERIFY__CONTAINER__UDOCKER_CONTAINER"] == "kub"
    assert exported["PHYSVERIFY__PATHS__DATA_DIR"] == str(sandbox.data)
    assert exported["PHYSVERIFY__RUNTIME__HEAVY_LOCK"].endswith("kubric.lock")
    assert exported["PHYSVERIFY__RUNTIME__LOCK_TIMEOUT_S"] == "0"
    assert all(exported[name] == "false" for name in DD_VARIABLES)
    log = sandbox.data / "runs" / "kubric-latest.log"
    assert log.is_symlink()
    assert log.read_text(encoding="utf-8").splitlines() == ["render: 0 launched", "render 2 done"]


def test_missing_container_is_pulled_and_created(sandbox: Sandbox) -> None:
    result = sandbox.run("smoke", INSPECT_EXIT="1", PHYSVERIFY_KUBRIC_PROFILE="fast")
    assert result.returncode == 0, result.stderr
    image = "kubricdockerhub/kubruntu:latest"
    assert sandbox.lines()[:3] == [
        "udocker inspect kub",
        f"udocker pull {image}",
        f"udocker create --name=kub {image}",
    ]
    assert sandbox.lines()[-1].endswith("dataset list --subset smoke")
    assert "--profile fast" in sandbox.lines()[-1]


def test_failed_build_still_lists_and_keeps_the_exit_code(sandbox: Sandbox) -> None:
    result = sandbox.run(BUILD_EXIT="1")
    assert result.returncode == 1
    assert sandbox.lines()[-1] == f"{PV}dataset list --subset full"


@pytest.mark.parametrize(
    "env",
    [
        {"IMPORT_EXIT": "1"},
        {"FREE_GB": "10"},
        {"DOCTOR_ENGINE": "docker"},
        {"PHYSVERIFY_KUBRIC_WORKERS": "0"},
        {"PHYSVERIFY_UDOCKER": "/nonexistent/udocker"},
    ],
    ids=["no-imports", "low-disk", "wrong-engine", "bad-workers", "no-udocker"],
)
def test_preflight_failures_stop_before_the_build(sandbox: Sandbox, env: dict[str, str]) -> None:
    result = sandbox.run(**env)
    assert result.returncode == 2
    assert not any("dataset build" in line for line in sandbox.lines())
