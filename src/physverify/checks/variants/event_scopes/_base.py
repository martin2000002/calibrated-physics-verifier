"""Event scope contract: a parameter-free test of whether a measured impact lies inside the
scope R3b image_relspeed declares; outside it the check does not apply."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

from physverify.checks.variants._image_impulse import Event
from physverify.config.plugin_context import PluginKind
from physverify.core.errors import ConfigError
from physverify.core.registry import Registry, discover
from physverify.schema.states import ObjectStateTable

EVENT_SCOPES_PACKAGE = "physverify.checks.variants.event_scopes"


@dataclass(frozen=True)
class SidedEvent:
    """An impact with the free rows of each side its line fits used; none when the check did not
    reach the sides (another body in contact)."""

    event: Event
    left: tuple[int, ...]
    right: tuple[int, ...]


@dataclass(frozen=True)
class ScopeLimits:
    """The check's side lengths in rows, its gap (native frames) and window margin (rows)."""

    side_min_rows: int
    side_max_rows: int
    gap_frames: float
    bound_margin_rows: int

    def reach_frames(self, step: int) -> float:
        """Native frames from an event to the far end of its longest side window."""
        return self.gap_frames + (self.side_max_rows + self.bound_margin_rows) * step


class EventScope(Protocol):
    """Registered by ``name``; constructed as ``cls()``. ``applies`` sees the states (with their
    primitives), the event with its sides and the clip's other events of the same level: never a
    role, a perturbation or a manifest."""

    name: ClassVar[str]

    def applies(
        self,
        states: ObjectStateTable,
        event: SidedEvent,
        others: Sequence[Event],
        gates: ScopeLimits,
    ) -> bool: ...


EVENT_SCOPES = PluginKind[type[EventScope]]("event_scopes")


def event_scopes() -> Registry[type[EventScope]]:
    return discover(EVENT_SCOPES_PACKAGE, (EventScope,), "name")


def build_scopes(
    names: Sequence[str], registry: Mapping[str, type[EventScope]]
) -> tuple[EventScope, ...]:
    """The scopes ``names`` in the given order; a ConfigError names the known ones."""
    unknown = [name for name in names if name not in registry]
    if unknown:
        known = ", ".join(sorted(registry))
        raise ConfigError(f"R3b event scopes {unknown} are unknown; known: {known}")
    return tuple(registry[name]() for name in names)


def failed_scope(
    scopes: Sequence[EventScope],
    states: ObjectStateTable,
    event: SidedEvent,
    others: Sequence[Event],
    gates: ScopeLimits,
) -> str | None:
    """The name of the first scope ``event`` lies outside of; None inside all of them."""
    return next((s.name for s in scopes if not s.applies(states, event, others, gates)), None)
