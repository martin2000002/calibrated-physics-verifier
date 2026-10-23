"""Per-invocation application context (no global state)."""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from physverify.app import plugins
from physverify.config.loader import CONFIG_DIR_ENV as CONFIG_DIR_ENV
from physverify.config.loader import default_config_dir as default_config_dir
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.config.models import is_device as is_device
from physverify.core.output import Echo
from physverify.core.registry import Registry
from physverify.runtime.container import ContainerEngine, make_runner
from physverify.runtime.heavy import LOCK_HELD_ENV, HeavyLock, NullLock


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


class AppContext:
    """Settings, lock, container runner and registries of one invocation, built lazily."""

    def __init__(
        self,
        invocation: Invocation,
        options: GlobalOptions,
        err: Echo | None = None,
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

    def docker(self) -> ContainerEngine:
        """The Kubric container runner of the configured engine (``[container]``)."""
        return make_runner(self.settings, os.getuid(), os.getgid())

    def registry(self, kind: str) -> Registry[Any]:
        if kind not in self._registries:
            self._registries[kind] = plugins.build(kind)
        return self._registries[kind]
