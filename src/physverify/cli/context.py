"""Per-invocation application context, injected through ``ctx.obj`` (no global state)."""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, NoReturn

import typer
from typer._click.globals import get_current_context

from physverify.checks._base import Check
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.core.errors import InvalidPluginError, NotImplementedCommandError, UsageError
from physverify.core.registry import Registry, discover, discover_modules
from physverify.datasets._base import DatasetSource
from physverify.diagnosis._base import StageMetric, metric_key
from physverify.extraction._base import StageProvider, provider_key, validate_provider
from physverify.runtime.docker import DockerRunner
from physverify.runtime.heavy import LOCK_HELD_ENV, HeavyLock, NullLock
from physverify.synth.families._base import SceneFamily
from physverify.synth.perturbations._base import (
    ScenePerturbation,
    TrajectoryPerturbation,
    validate_perturbation,
)
from physverify.timing._base import TimingTarget
from physverify.verification._base import VerdictPolicy

COMMANDS_PACKAGE = "physverify.cli.commands"
TASKS_PACKAGE = "physverify.kubric_worker.tasks"


@dataclass(frozen=True)
class GlobalOptions:
    config_dir: Path
    profile: str | None = None
    device: str | None = None
    overrides: tuple[str, ...] = ()
    no_lock: bool = False
    lock_timeout_s: float | None = None
    log_level: str = "WARNING"


@dataclass(frozen=True)
class Invocation:
    argv: tuple[str, ...]
    environ: Mapping[str, str]


def check_command(module: ModuleType) -> None:
    """A CLI plugin module defines NAME, OWNER, IMPLEMENTED and a typer ``app``."""
    ok = (
        isinstance(getattr(module, "NAME", None), str)
        and isinstance(getattr(module, "OWNER", None), str)
        and isinstance(getattr(module, "IMPLEMENTED", None), bool)
        and isinstance(getattr(module, "app", None), typer.Typer)
    )
    if not ok:
        raise InvalidPluginError(f"{module.__name__}: needs NAME, OWNER, IMPLEMENTED and app")


def command_name(module: ModuleType) -> str:
    name: str = module.NAME
    return name


def check_task(module: ModuleType) -> None:
    if not callable(getattr(module, "run", None)):
        raise InvalidPluginError(f"{module.__name__}: a container task defines run(job)")


def module_name(module: ModuleType) -> str:
    return module.__name__.rsplit(".", 1)[-1]


PLUGIN_KINDS: Mapping[str, Callable[[], Registry[Any]]] = MappingProxyType(
    {
        "checks": lambda: discover("physverify.checks", (Check,), "check_id"),
        "perturbations": lambda: discover(
            "physverify.synth.perturbations",
            (ScenePerturbation, TrajectoryPerturbation),
            "name",
            validate_perturbation,
        ),
        "families": lambda: discover("physverify.synth.families", (SceneFamily,), "name"),
        "providers": lambda: discover(
            "physverify.extraction.providers", (StageProvider,), provider_key, validate_provider
        ),
        "datasets": lambda: discover("physverify.datasets", (DatasetSource,), "name"),
        "container_tasks": lambda: discover_modules(TASKS_PACKAGE, check_task, module_name),
        "metrics": lambda: discover("physverify.diagnosis.metrics", (StageMetric,), metric_key),
        "policies": lambda: discover("physverify.verification.policies", (VerdictPolicy,), "name"),
        "timing_targets": lambda: discover("physverify.timing.targets", (TimingTarget,), "name"),
        "commands": lambda: discover_modules(COMMANDS_PACKAGE, check_command, command_name),
    }
)
"""Every plugin kind. Perceived and ground-truth providers share ``providers`` (keys
``<stage>/<name>`` and ``<stage>/ground_truth``)."""


class AppContext:
    """Settings, lock, container runner and registries of one CLI invocation, built lazily."""

    def __init__(
        self,
        invocation: Invocation,
        options: GlobalOptions,
        err: Callable[[str], None] | None = None,
    ) -> None:
        self.invocation = invocation
        self.options = options
        self._err = err or functools.partial(print, file=sys.stderr)
        self._registries: dict[str, Registry[Any]] = {}

    @cached_property
    def settings(self) -> Settings:
        return load_settings(
            self.options.config_dir,
            environ=self.invocation.environ,
            profile=self.options.profile,
            device=self.options.device,
            overrides=self.options.overrides,
        )

    def lock_disabled_reason(self) -> str | None:
        if self.invocation.environ.get(LOCK_HELD_ENV) == "1":
            return f"{LOCK_HELD_ENV}=1 (held by the caller)"
        if self.options.no_lock:
            return "--no-lock"
        if not self.settings.runtime.heavy_lock:
            return "runtime.heavy_lock is empty"
        return None

    def heavy_lock(self) -> HeavyLock | NullLock:
        reason = self.lock_disabled_reason()
        if reason is not None:
            return NullLock(reason)
        timeout = self.options.lock_timeout_s
        if timeout is None:
            timeout = self.settings.runtime.lock_timeout_s
        return HeavyLock(
            Path(self.settings.runtime.heavy_lock), timeout, self.invocation.argv, self._err
        )

    def docker(self) -> DockerRunner:
        settings = self.settings
        return DockerRunner(
            image=settings.runtime.docker_image,
            src_dir=settings.repo_root / "src",
            data_dir=settings.paths.data_dir,
            threads=settings.runtime.threads,
            uid=os.getuid(),
            gid=os.getgid(),
        )

    def registry(self, kind: str) -> Registry[Any]:
        if kind not in PLUGIN_KINDS:
            raise UsageError(f"unknown plugin kind {kind!r}; known: {sorted(PLUGIN_KINDS)}")
        if kind not in self._registries:
            self._registries[kind] = PLUGIN_KINDS[kind]()
        return self._registries[kind]


def app_context(ctx: typer.Context | None = None) -> AppContext:
    context = ctx if ctx is not None else get_current_context()
    found = context.find_object(AppContext)
    if found is None:
        raise RuntimeError("no AppContext: run commands through physverify.cli.main")
    return found


def heavy_command[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Run the command body while holding the heavy-job lock."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with app_context().heavy_lock():
            return func(*args, **kwargs)

    return wrapper


def not_implemented(owner: str, command: str) -> NoReturn:
    raise NotImplementedCommandError(f"{command}: not implemented (owned by cycle {owner})")
