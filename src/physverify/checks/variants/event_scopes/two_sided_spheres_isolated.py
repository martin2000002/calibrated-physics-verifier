"""An isolated impact of two spheres with enough free rows on both sides to fit each line.

``side_min_rows`` on each side: a line needs 2 parameters plus 1 degree of freedom for its
residual. Spheres: the image centre of a sphere's outline is its centre of mass whatever its spin;
that of a box or a cylinder moves with the spin, which the impact changes. Isolated: another event
within ``ScopeLimits.reach_frames`` shares rows with this one's longest side window."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from physverify.checks.variants._image_impulse import Event
from physverify.checks.variants.event_scopes._base import ScopeLimits, SidedEvent, both_spheres
from physverify.physics.tracks import row_step
from physverify.schema.states import ObjectStateTable


class TwoSidedSpheresIsolated:
    name: ClassVar[str] = "two_sided_spheres_isolated"

    def applies(
        self,
        states: ObjectStateTable,
        event: SidedEvent,
        others: Sequence[Event],
        gates: ScopeLimits,
    ) -> bool:
        return not self.failed(states, event, others, gates)

    def failed(
        self,
        states: ObjectStateTable,
        event: SidedEvent,
        others: Sequence[Event],
        gates: ScopeLimits,
    ) -> tuple[str, ...]:
        holds = {
            "two_sided": _two_sided(event, gates),
            "spheres": both_spheres(states, event.event),
            "isolated": _isolated(states, event.event, others, gates),
        }
        return tuple(name for name, ok in holds.items() if not ok)


def _two_sided(event: SidedEvent, gates: ScopeLimits) -> bool:
    return min(len(event.left), len(event.right)) >= gates.side_min_rows


def _isolated(
    states: ObjectStateTable, e: Event, others: Sequence[Event], gates: ScopeLimits
) -> bool:
    reach = gates.reach_frames(row_step(states.frame_index))
    return all(abs(o.t - e.t) > reach for o in others)


PLUGINS = (TwoSidedSpheresIsolated,)
