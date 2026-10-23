"""R4_missing truth: most missing frames of any object."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.magnitudes import nanmax, object_missing


class PermanenceTruth:
    check_id: ClassVar[str] = "R4_missing"

    def measure(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        tol: float,
        cuts: Sequence[int],
    ) -> float:
        return nanmax([object_missing(states, i) for i in range(states.n_objects)])


PLUGINS = (PermanenceTruth,)
