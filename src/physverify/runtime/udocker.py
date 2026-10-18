"""Runs container tasks with udocker in a named container created from the Kubric image, for
hosts without Docker or root (the H200 server; ADR 0003, section 4).

Unlike ``DockerRunner`` there is no ``--rm`` (the named container is reused), no ``--user`` (the
process runs as the calling host user) and no ``:ro`` (udocker does not honour it), so the
worker is told not to write ``.pyc`` files into the shared source tree.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from physverify.core.errors import ConfigError
from physverify.runtime.docker import (
    CONTAINER_DATA,
    CONTAINER_SRC,
    Runner,
    container_env,
    container_path,
    worker_command,
)

EXTRA_ENV = (("PYTHONDONTWRITEBYTECODE", "1"),)


@dataclass(frozen=True)
class UdockerRunner:
    bin: str
    container: str
    src_dir: Path
    data_dir: Path
    threads: int

    def __post_init__(self) -> None:
        if not self.container:
            raise ConfigError("container.udocker_container is empty")

    def container_path(self, host_path: Path) -> str:
        return container_path(self.data_dir, host_path)

    def argv(self, task: str, jobs: Path) -> list[str]:
        return self.command_argv(worker_command(task, self.container_path(jobs), self.threads))

    def command_argv(self, command: Sequence[str]) -> list[str]:
        env = [*container_env(self.threads), *EXTRA_ENV]
        return [
            self.bin, "run", "--nobanner",
            f"--volume={self.src_dir.resolve()}:{CONTAINER_SRC}",
            f"--volume={self.data_dir.resolve()}:{CONTAINER_DATA}",
            f"--workdir={CONTAINER_DATA}",
            *(f"--env={name}={value}" for name, value in env),
            self.container,
            *command,
        ]  # fmt: skip

    def probe_argv(self) -> list[str]:
        """Exits 0 when the named container exists."""
        return [self.bin, "inspect", self.container]

    def run(
        self, task: str, jobs: Path, runner: Runner = subprocess.run
    ) -> subprocess.CompletedProcess[bytes]:
        return runner(self.argv(task, jobs), check=False)
