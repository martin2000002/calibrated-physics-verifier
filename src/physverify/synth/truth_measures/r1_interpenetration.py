"""R1 truth: largest penetration over object pairs and object-floor, divided by the smaller L."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.magnitudes import floor_penetration, nanmax, penetration_between
from physverify.synth.perturbations._frame_windows import solid_at


class PenetrationTruth:
    check_id: ClassVar[str] = "R1"

    def measure(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        tol: float,
        cuts: Sequence[int],
    ) -> float:
        values: list[float] = []
        for t in range(states.n_frames):
            solids = [solid_at(states, i, t) for i in range(states.n_objects)]
            for i, solid in enumerate(solids):
                values.append(floor_penetration(solid))
                values.extend(penetration_between(solid, other) for other in solids[i + 1 :])
        return nanmax(values)


PLUGINS = (PenetrationTruth,)
