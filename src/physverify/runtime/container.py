"""The container engine selection point (``[container]`` section; ADR 0003, section 4).

A third engine is one module with a runner class, plus its name in ``ContainerSettings.engine``,
``ENGINES``, ``MISSING_HINTS`` and ``describe``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

from pydantic import Field

from physverify.config.models import FrozenModel, Settings
from physverify.runtime.docker import DockerRunner, Runner, preflight_command
from physverify.runtime.udocker import UdockerRunner

SECTION = "container"


class ContainerSettings(FrozenModel):
    engine: Literal["docker", "udocker"] = "docker"
    udocker_bin: str = "udocker"
    udocker_container: str = "kub"
    preflight_timeout_s: float = Field(default=300.0, gt=0)


class ContainerEngine(Protocol):
    """Runtime-side superset of ``synth.store.ContainerRunner``: the pipeline only uses
    ``run`` and ``container_path``; the argv methods serve checks and diagnostics."""

    def run(self, task: str, jobs: Path) -> subprocess.CompletedProcess[bytes]: ...

    def container_path(self, host_path: Path) -> str: ...

    def argv(self, task: str, jobs: Path) -> list[str]: ...

    def command_argv(self, command: Sequence[str]) -> list[str]: ...

    def probe_argv(self) -> list[str]: ...


EngineFactory = Callable[[Settings, ContainerSettings, int, int], ContainerEngine]


def _docker(settings: Settings, _: ContainerSettings, uid: int, gid: int) -> ContainerEngine:
    return DockerRunner(
        image=settings.runtime.docker_image,
        src_dir=settings.repo_root / "src",
        data_dir=settings.paths.data_dir,
        threads=settings.runtime.threads,
        uid=uid,
        gid=gid,
    )


def _udocker(
    settings: Settings, section: ContainerSettings, _uid: int, _gid: int
) -> ContainerEngine:
    return UdockerRunner(
        bin=section.udocker_bin,
        container=section.udocker_container,
        src_dir=settings.repo_root / "src",
        data_dir=settings.paths.data_dir,
        threads=settings.runtime.threads,
    )


ENGINES: Mapping[str, EngineFactory] = MappingProxyType({"docker": _docker, "udocker": _udocker})
MISSING_HINTS: Mapping[str, str] = MappingProxyType(
    {
        "docker": "run `physverify dataset pull-image`",
        "udocker": "run `scripts/h200/run_kubric.sh` (it pulls and creates the container)",
    }
)


def container_settings(settings: Settings) -> ContainerSettings:
    return settings.section(SECTION, ContainerSettings)


def make_runner(settings: Settings, uid: int, gid: int) -> ContainerEngine:
    section = container_settings(settings)
    return ENGINES[section.engine](settings, section, uid, gid)


def describe(settings: Settings) -> str:
    """One line for ``physverify doctor``."""
    section = container_settings(settings)
    if section.engine == "udocker":
        return f"udocker ({section.udocker_container} via {section.udocker_bin})"
    return section.engine


def missing_reason(
    settings: Settings, runner: ContainerEngine, run: Runner = subprocess.run
) -> str | None:
    """None when the engine can start the Kubric container, else why not and what to do."""
    engine = container_settings(settings).engine
    probe = runner.probe_argv()
    try:
        result: Any = run(probe, capture_output=True, check=False)
    except OSError:
        return f"{engine}: {probe[0]} cannot run; {MISSING_HINTS[engine]}"
    if result.returncode == 0:
        return None
    return f"{engine}: `{' '.join(probe)}` failed; {MISSING_HINTS[engine]}"


def preflight_problem(
    runner: ContainerEngine, timeout_s: float, run: Runner = subprocess.run
) -> str | None:
    """None when the container imports the build's modules with the build's mounts and workdir."""
    argv = runner.command_argv(preflight_command())
    try:
        result: Any = run(argv, check=False, timeout=timeout_s)
    except OSError as exc:
        return f"{argv[0]} cannot run: {exc}"
    except subprocess.TimeoutExpired:
        return f"`{' '.join(argv)}` did not finish in {timeout_s} s"
    if result.returncode == 0:
        return None
    return f"`{' '.join(argv)}` exited {result.returncode}"
