"""Runs container tasks in the pinned Kubric image (ADR 0003, section 4)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from physverify.core.errors import ConfigError, UsageError

CONTAINER_SRC = "/app/src"
CONTAINER_DATA = PurePosixPath("/data")
WORKER_MODULE = "physverify.kubric_worker"
PREFLIGHT_IMPORTS = ("kubric", "bpy", "pybullet", f"{WORKER_MODULE}.build_scene")

Runner = Callable[..., "subprocess.CompletedProcess[bytes]"]


def container_path(data_dir: Path, host_path: Path) -> str:
    """Host path under ``data_dir`` as seen inside the container (``/data/...``)."""
    try:
        relative = host_path.resolve().relative_to(data_dir.resolve())
    except ValueError as exc:
        raise UsageError(f"{host_path} is outside the data directory {data_dir}") from exc
    return str(CONTAINER_DATA.joinpath(*relative.parts))


def container_env(threads: int) -> list[tuple[str, str]]:
    """Environment every engine passes to the worker process."""
    return [
        ("PYTHONPATH", CONTAINER_SRC),
        ("HOME", "/tmp"),
        ("OMP_NUM_THREADS", str(threads)),
    ]


def image_present(image: str, run: Runner = subprocess.run) -> bool:
    return (
        run(["docker", "image", "inspect", image], capture_output=True, check=False).returncode == 0
    )


def worker_command(task: str, jobs_container_path: str, threads: int) -> list[str]:
    return [
        "python3", "-m", WORKER_MODULE, task,
        "--jobs", jobs_container_path,
        "--threads", str(threads),
    ]  # fmt: skip


def preflight_command() -> list[str]:
    """Imports what a build needs, from the mounted source tree as the worker sees it."""
    return ["python3", "-c", f"import {', '.join(PREFLIGHT_IMPORTS)}"]


@dataclass(frozen=True)
class DockerRunner:
    image: str
    src_dir: Path
    data_dir: Path
    threads: int
    uid: int
    gid: int

    def container_path(self, host_path: Path) -> str:
        return container_path(self.data_dir, host_path)

    def argv(self, task: str, jobs: Path) -> list[str]:
        return self.command_argv(worker_command(task, self.container_path(jobs), self.threads))

    def command_argv(self, command: Sequence[str]) -> list[str]:
        if not self.image:
            raise ConfigError("runtime.docker_image is empty")
        env = [
            arg for name, value in container_env(self.threads) for arg in ("-e", f"{name}={value}")
        ]
        return [
            "docker", "run", "--rm",
            "--user", f"{self.uid}:{self.gid}",
            "-v", f"{self.src_dir.resolve()}:{CONTAINER_SRC}:ro",
            "-v", f"{self.data_dir.resolve()}:{CONTAINER_DATA}",
            "-w", str(CONTAINER_DATA),
            *env,
            self.image,
            *command,
        ]  # fmt: skip

    def probe_argv(self) -> list[str]:
        """Exits 0 when the image is available locally."""
        return ["docker", "image", "inspect", self.image]

    def run(
        self, task: str, jobs: Path, runner: Runner = subprocess.run
    ) -> subprocess.CompletedProcess[bytes]:
        return runner(self.argv(task, jobs), check=False)
