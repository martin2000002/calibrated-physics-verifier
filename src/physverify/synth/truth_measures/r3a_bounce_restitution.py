"""R3a truth: largest restitution excess (e - 1) over the floor impacts of spheres."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.magnitudes import bounce_restitution, nanmax


class RestitutionTruth:
    check_id: ClassVar[str] = "R3a"

    def measure(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        tol: float,
        cuts: Sequence[int],
    ) -> float:
        g = np.asarray(scene.gravity, dtype=np.float64)
        ids = states.object_ids.astype(str).tolist()
        values = [
            bounce_restitution(states, ids.index(r.object_a), r.t_start, g) - 1.0
            for r in contacts.records
            if r.kind is ContactKind.IMPACT
            and r.object_b == ENV_ID
            and str(states.primitive[ids.index(r.object_a)]) == PrimitiveKind.SPHERE.value
        ]
        return nanmax(values)


PLUGINS = (RestitutionTruth,)
