"""R4_size truth: largest size change ratio of any object."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from physverify.physics.measures.object_integrity import size_change_ratio
from physverify.schema.contacts import ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.magnitudes import nanmax


class SizeConstancyTruth:
    check_id: ClassVar[str] = "R4_size"

    def measure(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        tol: float,
        cuts: Sequence[int],
    ) -> float:
        return nanmax([size_change_ratio(states.size[:, i]) for i in range(states.n_objects)])


PLUGINS = (SizeConstancyTruth,)
