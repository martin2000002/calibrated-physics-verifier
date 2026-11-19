"""R3b truth: largest kinetic-energy gain over the impacts between two objects."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.magnitudes import nanmax, native_fps, pair_energy_gain


class EnergyGainTruth:
    check_id: ClassVar[str] = "R3b"

    def measure(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        tol: float,
        cuts: Sequence[int],
    ) -> float:
        g = np.asarray(scene.gravity, dtype=np.float64)
        masses = {obj.object_id: obj.mass for obj in scene.objects}
        ids = states.object_ids.astype(str).tolist()
        fps = native_fps(states)
        values = [
            pair_energy_gain(
                states,
                contacts,
                (ids.index(r.object_a), ids.index(r.object_b)),
                masses[r.object_a] / masses[r.object_b],
                r.t_start / fps,
                g,
                tol,
            )
            for r in contacts.records
            if r.kind is ContactKind.IMPACT and r.object_b != ENV_ID
        ]
        return nanmax(values)


PLUGINS = (EnergyGainTruth,)
