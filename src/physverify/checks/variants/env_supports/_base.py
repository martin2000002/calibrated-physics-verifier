"""ENV support contract: a rule that lets the ENV records overlapping an impact stop counting as
external contacts of the pair (and as bounds of its side rows)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol

from physverify.checks.r3b_collision_energy_gain import external_contacts
from physverify.checks.variants._image_impulse import Event, as_record, env_records
from physverify.config.plugin_context import PluginKind
from physverify.core.errors import ConfigError
from physverify.core.registry import Registry, discover
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable
from physverify.schema.states import ObjectStateTable

ENV_SUPPORTS_PACKAGE = "physverify.checks.variants.env_supports"

Supported = tuple[tuple[ContactRecord, ...], dict[str, Any]]
"""The ENV records an event's rules hold and its ``floor_supports`` detail."""
Sides = Callable[[Sequence[ContactRecord]], tuple[list[int], list[int]]]
"""The before and after rows of an event bounded by the given records."""
Rows = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class SupportQuery:
    """One impact and the ENV records of its bodies that a rule may hold.

    ``records`` are first the ``named`` ENV records that ``external_contacts`` names for the
    event (``named_ids`` its names for them), which must all be held, then the other ENV bounds
    of either body, which only bound the side rows. ``sides_without`` walks the before and after
    rows with the given ``records`` indices removed from the bounds; ``others`` are the bounds
    that are not ENV records (the event's own excluded), with the sides' ``margin``."""

    event: Event
    pair: tuple[int, int]
    records: tuple[ContactRecord, ...]
    named: int
    named_ids: tuple[str, ...]
    others: tuple[ContactRecord, ...]
    sides_without: Callable[[Sequence[int]], Rows]
    margin: int
    max_tilt_deg: float


@dataclass(frozen=True)
class Support:
    """The indices of the ``records`` a rule holds, and what it measured to hold them."""

    records: tuple[int, ...]
    angles: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SupportGeometry:
    """The check's side margin (rows), the rules' tilt tolerance (deg) and the declared bias
    budget a supported event's s carries."""

    margin: int
    max_tilt_deg: float
    bias_budget: float


class SupportRule(Protocol):
    """Registered by ``name``; constructed as ``cls()``. ``allows`` returns None when it holds
    none of the records."""

    name: ClassVar[str]

    def allows(self, states: ObjectStateTable, query: SupportQuery) -> Support | None: ...


ENV_SUPPORTS = PluginKind[type[SupportRule]]("env_supports")


def env_supports() -> Registry[type[SupportRule]]:
    return discover(ENV_SUPPORTS_PACKAGE, (SupportRule,), "name")


def build_rules(
    names: Sequence[str], registry: Mapping[str, type[SupportRule]]
) -> tuple[SupportRule, ...]:
    """The rules ``names`` in the given order; a ConfigError names the known ones."""
    unknown = [name for name in names if name not in registry]
    if unknown:
        known = ", ".join(sorted(registry))
        raise ConfigError(f"R3b env supports {unknown} are unknown; known: {known}")
    return tuple(registry[name]() for name in names)


def supports(
    states: ObjectStateTable,
    contacts: ContactTable | None,
    event: Event,
    bounds: Sequence[ContactRecord],
    rules: Sequence[SupportRule],
    sides: Sides,
    geometry: SupportGeometry,
) -> Supported | None:
    """The ENV records the rules together hold for ``event`` and their detail, when they hold
    every record ``external_contacts`` names (all or nothing per event); None otherwise."""
    record = as_record(event)
    named = [
        (r, ids[0])
        for r in env_records(contacts)
        if (ids := external_contacts(states, table_of(states, [r]), record))
    ]
    if not named:
        return None
    first = tuple(r for r, _ in named)
    bodies = (event.a, event.b)
    nearby = [r for r in without(bounds, first) if r.object_b == ENV_ID and r.object_a in bodies]
    records = (*first, *nearby)

    def sides_without(held: Sequence[int]) -> Rows:
        left, right = sides(without(bounds, [records[i] for i in held]))
        return tuple(left), tuple(right)

    query = SupportQuery(
        event=event,
        pair=(states.object_index(event.a), states.object_index(event.b)),
        records=records,
        named=len(first),
        named_ids=tuple(name for _, name in named),
        others=tuple(r for r in bounds if r.object_b != ENV_ID),
        sides_without=sides_without,
        margin=geometry.margin,
        max_tilt_deg=geometry.max_tilt_deg,
    )
    held = held_by(states, query, rules)
    if held is None:
        return None
    indices, angles = held
    detail = {
        "records": list(query.named_ids),
        "held_nearby": len(indices) - query.named,
        "angles": angles,
        "bias_budget": geometry.bias_budget,
    }
    return tuple(records[i] for i in indices), detail


def held_by(
    states: ObjectStateTable, query: SupportQuery, rules: Sequence[SupportRule]
) -> tuple[list[int], dict[str, float]] | None:
    """The sorted ``records`` indices the rules together hold and their angles, when they hold
    every named one; None otherwise."""
    held: set[int] = set()
    angles: dict[str, float] = {}
    for rule in rules:
        found = rule.allows(states, query)
        if found is not None:
            held |= set(found.records)
            angles |= {f"{rule.name}.{k}": v for k, v in found.angles.items()}
    if not set(range(query.named)) <= held:
        return None
    return sorted(held), angles


def table_of(states: ObjectStateTable, records: Sequence[ContactRecord]) -> ContactTable:
    return ContactTable(
        states.clip_id, states.frame, states.unit, states.source_key, tuple(records)
    )


def without(records: Sequence[ContactRecord], held: Sequence[ContactRecord]) -> list[ContactRecord]:
    """``records`` but the very objects of ``held``."""
    return [r for r in records if not any(r is h for h in held)]


def without_resting(contacts: ContactTable) -> ContactTable:
    """The table without its ENV RESTING records: where a supported event may be detected."""
    kept = [
        r for r in contacts.records if r.object_b != ENV_ID or r.kind is not ContactKind.RESTING
    ]
    return replace(contacts, records=tuple(kept))
