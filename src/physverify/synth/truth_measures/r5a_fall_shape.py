"""R5a truth: inconsistency of the accelerations fitted to every flight of every object."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.magnitudes import R5A_MIN_FRAMES, flight_fits, inconsistency


class GravityConsistencyTruth:
    check_id: ClassVar[str] = "R5a"

    def measure(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        tol: float,
        cuts: Sequence[int],
    ) -> float:
        fits = [
            fit
            for i in range(states.n_objects)
            for fit in flight_fits(states, contacts, i, R5A_MIN_FRAMES, cuts)
        ]
        return inconsistency(fits)


PLUGINS = (GravityConsistencyTruth,)
