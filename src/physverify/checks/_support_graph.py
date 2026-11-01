"""Resting events of R2: rest runs, support components and the records that link them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.physics.tracks import contact_rows, rest_mask, segments
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable
from physverify.schema.states import ObjectStateTable

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Event:
    """One component of resting bodies over a run of rows with a constant resting set."""

    start: int
    stop: int
    bodies: tuple[int, ...]  # table indices, sorted by object id
    support: tuple[ContactRecord, ...]  # RESTING records inside the component or with ENV
    excluded: tuple[ContactRecord, ...]  # other records touching the component at the row

    @property
    def row(self) -> int:
        return (self.start + self.stop - 1) // 2


def resting_mask(states: ObjectStateTable, min_rows: int, rest_speed: float) -> BoolArray:
    """Rows x bodies at rest for at least ``min_rows`` consecutive rows."""
    resting = np.zeros(states.visible.shape, dtype=np.bool_)
    for i in range(states.n_objects):
        mask = rest_mask(states, i, rest_speed)
        for start, stop in segments(mask, min_rows):
            resting[start:stop, i] = True
    return resting


def agents_in(states: ObjectStateTable, event: Event, agent_labels: Sequence[str]) -> list[str]:
    """Sorted agent labels among the bodies of ``event`` and the bodies its records touch."""
    records = event.support + event.excluded
    involved = {str(states.object_ids[i]) for i in event.bodies}
    involved |= {r.object_a for r in records} | {r.object_b for r in records}
    labels = {str(states.labels[states.object_index(o)]) for o in involved - {ENV_ID}}
    return sorted(labels & set(agent_labels))


def runs(resting: BoolArray) -> list[tuple[int, int]]:
    """Maximal [start, stop) row runs with a constant, non-empty set of resting bodies."""
    found: list[tuple[int, int]] = []
    start = 0
    for row in range(1, resting.shape[0] + 1):
        if row == resting.shape[0] or not np.array_equal(resting[row], resting[start]):
            if bool(resting[start].any()):
                found.append((start, row))
            start = row
    return found


def components(
    states: ObjectStateTable,
    contacts: ContactTable,
    resting: BoolArray,
    start: int,
    stop: int,
) -> list[Event]:
    """Connected groups of resting bodies linked by RESTING records active at the central row."""
    row = (start + stop - 1) // 2
    fi = states.frame_index
    members = {str(states.object_ids[i]) for i in np.flatnonzero(resting[start])}
    active = [r for r in contacts.records if bool(contact_rows(r, fi)[row])]
    links = [r for r in active if is_support(r, members)]
    parent = {m: m for m in members}

    def root(node: str) -> str:
        while parent[node] != node:
            node = parent[node]
        return node

    for record in links:
        if record.object_b != ENV_ID:
            parent[root(record.object_a)] = root(record.object_b)
    groups: dict[str, list[str]] = {}
    for member in sorted(members):
        groups.setdefault(root(member), []).append(member)
    return [group_event(states, active, links, group, start, stop) for group in groups.values()]


def is_support(record: ContactRecord, members: set[str]) -> bool:
    ends_resting = record.object_a in members and record.object_b in members | {ENV_ID}
    return record.kind is ContactKind.RESTING and ends_resting


def group_event(
    states: ObjectStateTable,
    active: Sequence[ContactRecord],
    links: Sequence[ContactRecord],
    group: list[str],
    start: int,
    stop: int,
) -> Event:
    names = set(group)
    support = tuple(r for r in links if r.object_a in names)
    used = {id(r) for r in support}
    excluded = tuple(r for r in active if id(r) not in used and names & {r.object_a, r.object_b})
    bodies = tuple(states.object_index(name) for name in group)
    return Event(start, stop, bodies, support, excluded)


def reaches_ground(event: Event) -> bool:
    """The component is connected, so one support on the environment links every body to it."""
    return any(r.object_b == ENV_ID for r in event.support)
