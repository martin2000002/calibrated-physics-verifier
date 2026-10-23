"""Ground-truth magnitude contract: one plugin per check, keyed by the check's ``check_id``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, Protocol, runtime_checkable

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable


@runtime_checkable
class TruthMeasure(Protocol):
    """Registered by ``check_id``; instantiated without arguments. ``cuts`` are frames where
    gravity is known to change, so R5 fits each side of them separately (the windowing of
    ``r5a_gravity_switch.measure``)."""

    check_id: ClassVar[str]

    def measure(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        tol: float,
        cuts: Sequence[int],
    ) -> float: ...


def measure_clip(
    registry: Mapping[str, type[TruthMeasure]],
    scene: SceneSpec,
    states: ObjectStateTable,
    contacts: ContactTable,
    tol: float,
    cuts: Sequence[int] = (),
) -> dict[str, float]:
    """Every registered magnitude of the clip, by check id."""
    return {
        check_id: plugin().measure(scene, states, contacts, tol, cuts)
        for check_id, plugin in registry.items()
    }
