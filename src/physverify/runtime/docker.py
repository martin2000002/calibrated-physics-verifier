"""Runs container tasks in the pinned Kubric image (ADR 0003, section 4)."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from physverify.core.errors import ConfigError, UsageError

CONTAINER_SRC = "/app/src"
CONTAINER_DATA = PurePosixPath("/data")
WORKER_MODULE = "physverify.kubric_worker"

Runner = Callable[..., "subprocess.CompletedProcess[bytes]"]


@dataclass(frozen=True)
class DockerRunner:
    image: str
    src_dir: Path
    data_dir: Path
    threads: int
    uid: int
    gid: int

    def container_path(self, host_path: Path) -> str:
        """Host path under ``data_dir`` as seen inside the container (``/data/...``)."""
        try:
            relative = host_path.resolve().relative_to(self.data_dir.resolve())
        except ValueError as exc:
            raise UsageError(f"{host_path} is outside the data directory {self.data_dir}") from exc
        return str(CONTAINER_DATA.joinpath(*relative.parts))

    def argv(self, task: str, jobs: Path) -> list[str]:
        if not self.image:
            raise ConfigError("runtime.docker_image is empty")
        return [
            "docker", "run", "--rm",
            "--user", f"{self.uid}:{self.gid}",
            "-v", f"{self.src_dir.resolve()}:{CONTAINER_SRC}:ro",
            "-v", f"{self.data_dir.resolve()}:{CONTAINER_DATA}",
            "-w", str(CONTAINER_DATA),
            "-e", f"PYTHONPATH={CONTAINER_SRC}",
            "-e", "HOME=/tmp",
            "-e", f"OMP_NUM_THREADS={self.threads}",
            self.image,
            "python3", "-m", WORKER_MODULE, task,
            "--jobs", self.container_path(jobs),
            "--threads", str(self.threads),
        ]  # fmt: skip

    def run(
        self, task: str, jobs: Path, runner: Runner = subprocess.run
    ) -> subprocess.CompletedProcess[bytes]:
        return runner(self.argv(task, jobs), check=False)
