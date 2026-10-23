"""scripts/h200/run_kubric.sh against fake ``docker``, ``udocker``, ``uv`` and ``df``."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from shell_sandbox import Sandbox, executable

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "h200"
SCRIPT_NAMES = ("_common.sh", "run_kubric.sh")
TOOLS = (
    "bash", "basename", "cat", "date", "dirname", "env", "grep", "ln", "mkdir", "nproc", "sed",
    "sort", "tail", "tee", "tr",
)  # fmt: skip
PV = (
    "uv run --no-sync physverify --profile precise --set runtime.threads=16 "
    "--set dataset.render.workers=12 --set dataset.render.jobs_per_process=8 "
)
IMAGE = "kubric@sha256:pinned"
LATEST = "kubricdockerhub/kubruntu:latest"
PROBE = f"{PV}doctor --probe-container"
DD_VARIABLES = (
    "DD_TRACE_ENABLED",
    "DD_SYMBOL_DATABASE_UPLOAD_ENABLED",
    "DD_INSTRUMENTATION_TELEMETRY_ENABLED",
    "DD_REMOTE_CONFIGURATION_ENABLED",
)

FAKE_UV = """#!/bin/bash
printf 'uv %s\\n' "$*" >> "$CALLS"
env | grep -E '^(PHYSVERIFY__|DD_|FROM_ENV_FILE)' | sort > "$ENV_DUMP"
case "$*" in
  *" doctor")
    echo "docker_image: $IMAGE"
    echo "container_engine: ${DOCTOR_ENGINE:-$PHYSVERIFY__CONTAINER__ENGINE (detail)}" ;;
  *" doctor --probe-container") exit "${PROBE_EXIT:-0}" ;;
  *" synth build "*) echo "render: 0 launched"; exit "${BUILD_EXIT:-0}" ;;
  *" synth list "*) echo "render 2 done" ;;
esac
"""
FAKE_DOCKER = """#!/bin/bash
printf 'docker %s\\n' "$*" >> "$CALLS"
case "$1" in
  info) exit "${DOCKER_INFO_EXIT:-0}" ;;
  image) exit "${IMAGE_EXIT:-0}" ;;
  run) exit "${IMPORT_EXIT:-0}" ;;
esac
"""
FAKE_UDOCKER = """#!/bin/bash
printf 'udocker %s\\n' "$*" >> "$CALLS"
case "$1" in
  inspect) exit "${INSPECT_EXIT:-0}" ;;
  ps)
    echo "CONTAINER ID                         P M NAMES              IMAGE"
    [[ -n "${UNLISTED:-}" ]] || echo "0f1e-uuid . W ['kub']   ${CONTAINER_IMAGE:-$IMAGE}" ;;
esac
"""
FAKE_DF = """#!/bin/bash
printf ' Avail\\n %s\\n' "${FREE_GB:-500}"
"""
UDOCKER = {"PHYSVERIFY_KUBRIC_ENGINE": "udocker"}


class KubricSandbox(Sandbox):
    """A copied ``scripts/h200`` whose H200 root is the temporary directory."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, SCRIPT_NAMES, TOOLS)
        self.data = root / "kubric-data"
        self.env_dump = root / "env.txt"
        self.udocker = root / "udocker-venv" / "bin" / "udocker"
        self.udocker.parent.mkdir(parents=True)
        executable(self.fakes / "uv", FAKE_UV)
        executable(self.fakes / "docker", FAKE_DOCKER)
        executable(self.udocker, FAKE_UDOCKER)
        executable(self.fakes / "df", FAKE_DF)
        self.env.update(
            {"ENV_DUMP": str(self.env_dump), "IMAGE": IMAGE, "PHYSVERIFY_H200_ROOT": str(root)}
        )

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return self.call("run_kubric.sh", *args, **env)

    def exported(self) -> dict[str, str]:
        pairs = (line.split("=", 1) for line in self.env_dump.read_text().splitlines())
        return dict(pairs)


@pytest.fixture
def sandbox(tmp_path: Path) -> KubricSandbox:
    return KubricSandbox(tmp_path)


def test_script_parses_and_is_executable() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    for name in (*SCRIPT_NAMES, "env.example.sh"):
        subprocess.run([bash, "-n", str(SCRIPTS / name)], check=True)
    assert os.access(SCRIPTS / "run_kubric.sh", os.X_OK)


def test_docker_is_the_default_engine(sandbox: KubricSandbox) -> None:
    result = sandbox.run()
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [
        f"{PV}doctor",
        "docker info",
        f"{PV}doctor",
        f"docker image inspect {IMAGE}",
        PROBE,
        f"{PV}synth build --subset full",
        f"{PV}synth list --subset full",
    ]
    exported = sandbox.exported()
    assert exported["PHYSVERIFY__CONTAINER__ENGINE"] == "docker"
    assert "PHYSVERIFY__CONTAINER__UDOCKER_BIN" not in exported
    assert all(exported[name] == "false" for name in DD_VARIABLES)


def test_paths_derive_from_the_h200_root(sandbox: KubricSandbox) -> None:
    (sandbox.root / "env.sh").write_text("export FROM_ENV_FILE=1\n", encoding="utf-8")
    assert sandbox.run().returncode == 0
    exported = sandbox.exported()
    assert exported["FROM_ENV_FILE"] == "1"
    assert exported["PHYSVERIFY__PATHS__DATA_DIR"] == str(sandbox.data)
    assert exported["PHYSVERIFY__RUNTIME__HEAVY_LOCK"] == str(sandbox.root / "kubric.lock")
    assert exported["PHYSVERIFY__RUNTIME__LOCK_TIMEOUT_S"] == "0"
    log = sandbox.data / "runs" / "kubric-latest.log"
    assert log.is_symlink()
    assert log.read_text(encoding="utf-8").splitlines() == [
        "container_engine: docker",
        f"container_image: {IMAGE}",
        "render: 0 launched",
        "render 2 done",
    ]


def test_overrides_win_over_the_root(sandbox: KubricSandbox, tmp_path: Path) -> None:
    data, lock = tmp_path / "other-data", tmp_path / "other.lock"
    env = {"PHYSVERIFY__PATHS__DATA_DIR": str(data), "PHYSVERIFY_KUBRIC_LOCK": str(lock)}
    assert sandbox.run(**env).returncode == 0
    exported = sandbox.exported()
    assert exported["PHYSVERIFY__PATHS__DATA_DIR"] == str(data)
    assert exported["PHYSVERIFY__RUNTIME__HEAVY_LOCK"] == str(lock)
    assert (data / "runs" / "kubric-latest.log").is_symlink()


def test_missing_image_is_pulled(sandbox: KubricSandbox) -> None:
    assert sandbox.run(IMAGE_EXIT="1").returncode == 0
    assert sandbox.lines()[3:6] == [
        f"docker image inspect {IMAGE}",
        f"{PV}synth pull-image",
        PROBE,
    ]


def test_udocker_engine(sandbox: KubricSandbox) -> None:
    result = sandbox.run(**UDOCKER)
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == [
        f"{PV}doctor",
        "udocker inspect kub",
        "udocker ps",
        PROBE,
        f"{PV}synth build --subset full",
        f"{PV}synth list --subset full",
    ]
    exported = sandbox.exported()
    assert exported["PHYSVERIFY__CONTAINER__ENGINE"] == "udocker"
    assert exported["PHYSVERIFY__CONTAINER__UDOCKER_BIN"] == str(sandbox.udocker)
    assert exported["PHYSVERIFY__CONTAINER__UDOCKER_CONTAINER"] == "kub"
    log = sandbox.data / "runs" / "kubric-latest.log"
    assert log.read_text(encoding="utf-8").splitlines()[:2] == [
        "container_engine: udocker",
        f"container_image: {IMAGE}",
    ]


def test_missing_udocker_container_is_created_from_the_pinned_digest(
    sandbox: KubricSandbox,
) -> None:
    result = sandbox.run("smoke", INSPECT_EXIT="1", PHYSVERIFY_KUBRIC_PROFILE="fast", **UDOCKER)
    assert result.returncode == 0, result.stderr
    assert sandbox.lines()[1:5] == [
        "udocker inspect kub",
        f"udocker pull {IMAGE}",
        f"udocker create --name=kub {IMAGE}",
        "udocker ps",
    ]
    assert sandbox.lines()[-1].endswith("synth list --subset smoke")
    assert "--profile fast" in sandbox.lines()[-1]


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"CONTAINER_IMAGE": LATEST}, f"comes from {LATEST}, not {IMAGE}"),
        ({"UNLISTED": "1"}, "cannot verify the image"),
    ],
    ids=["other-image", "unverifiable"],
)
def test_an_existing_container_off_the_pin_is_refused(
    sandbox: KubricSandbox, env: dict[str, str], message: str
) -> None:
    result = sandbox.run(**env, **UDOCKER)
    assert result.returncode == 2
    assert message in result.stderr and "udocker rm kub" in result.stderr
    assert not any("build" in line or "--probe-container" in line for line in sandbox.lines())


def test_an_explicit_image_accepts_its_container(sandbox: KubricSandbox) -> None:
    result = sandbox.run(CONTAINER_IMAGE=LATEST, PHYSVERIFY_KUBRIC_IMAGE=LATEST, **UDOCKER)
    assert result.returncode == 0, result.stderr
    log = sandbox.data / "runs" / "kubric-latest.log"
    assert f"container_image: {LATEST}" in log.read_text(encoding="utf-8").splitlines()


def test_failed_build_still_lists_and_keeps_the_exit_code(sandbox: KubricSandbox) -> None:
    result = sandbox.run(BUILD_EXIT="1")
    assert result.returncode == 1
    assert sandbox.lines()[-1] == f"{PV}synth list --subset full"


@pytest.mark.parametrize(
    "env",
    [
        {"PROBE_EXIT": "1"},
        {"DOCKER_INFO_EXIT": "1"},
        {"FREE_GB": "10"},
        {"DOCTOR_ENGINE": "udocker"},
        {"PHYSVERIFY_KUBRIC_WORKERS": "0"},
        {"PHYSVERIFY_KUBRIC_ENGINE": "podman"},
        {"PROBE_EXIT": "1", **UDOCKER},
        {"DOCTOR_ENGINE": "docker", **UDOCKER},
        {"PHYSVERIFY_UDOCKER": "/nonexistent/udocker", **UDOCKER},
    ],
    ids=[
        "no-imports", "no-docker", "low-disk", "wrong-engine", "bad-workers", "bad-engine",
        "udocker-no-imports", "udocker-wrong-engine", "no-udocker",
    ],
)  # fmt: skip
def test_preflight_failures_stop_before_the_build(
    sandbox: KubricSandbox, env: dict[str, str]
) -> None:
    result = sandbox.run(**env)
    assert result.returncode == 2
    assert not any("synth build" in line for line in sandbox.lines())
