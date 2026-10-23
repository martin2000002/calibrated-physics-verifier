"""What a service reads from the application context: settings and plugin registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from physverify.config.models import Settings
from physverify.core.registry import Registry


class PluginContext(Protocol):
    """The settings and plugin registries of one invocation."""

    @property
    def settings(self) -> Settings: ...

    def registry(self, kind: str) -> Registry[Any]: ...


@dataclass(frozen=True)
class PluginKind[T]:
    """A registry name tied to the type of its entries; each kind is declared beside its
    protocol."""

    name: str


def registry_of[T](context: PluginContext, kind: PluginKind[T]) -> Registry[T]:
    """The registry of ``kind``, typed by its declaration."""
    registry: Registry[T] = context.registry(kind.name)
    return registry
