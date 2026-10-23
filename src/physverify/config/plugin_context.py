"""What a service reads from the application context: settings and plugin registries."""

from __future__ import annotations

from typing import Any, Protocol

from physverify.config.models import Settings
from physverify.core.registry import Registry


class PluginContext(Protocol):
    """The settings and plugin registries of one invocation."""

    @property
    def settings(self) -> Settings: ...

    def registry(self, kind: str) -> Registry[Any]: ...
