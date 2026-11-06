"""Event gate contract: a parameter-free test that keeps or drops a detected image impulse event."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, Protocol

from physverify.checks.variants._image_impulse import Candidate, Gates
from physverify.config.plugin_context import PluginKind
from physverify.core.errors import ConfigError
from physverify.core.registry import Registry, discover
from physverify.schema.states import ObjectStateTable

EVENT_GATES_PACKAGE = "physverify.checks.variants.event_gates"


class EventGate(Protocol):
    """Registered by ``name``; constructed as ``cls()``. ``accepts`` sees the merged candidate of
    ``detect`` with the side rows its line fits used."""

    name: ClassVar[str]

    def accepts(self, states: ObjectStateTable, event: Candidate, gates: Gates) -> bool: ...


EVENT_GATES = PluginKind[type[EventGate]]("event_gates")


def event_gates() -> Registry[type[EventGate]]:
    return discover(EVENT_GATES_PACKAGE, (EventGate,), "name")


def build_gates(
    names: Sequence[str], registry: Mapping[str, type[EventGate]]
) -> tuple[EventGate, ...]:
    """The gates ``names`` in the given order; a ConfigError names the known ones."""
    unknown = [name for name in names if name not in registry]
    if unknown:
        known = ", ".join(sorted(registry))
        raise ConfigError(f"R3b event gates {unknown} are unknown; known: {known}")
    return tuple(registry[name]() for name in names)
