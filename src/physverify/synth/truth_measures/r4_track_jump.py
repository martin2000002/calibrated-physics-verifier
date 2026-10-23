"""R4_jump truth: largest jump ratio of any object outside its impact frames."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.magnitudes import nanmax, object_jump


class TrackJumpTruth:
    check_id: ClassVar[str] = "R4_jump"

    def measure(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        tol: float,
        cuts: Sequence[int],
    ) -> float:
        return nanmax([object_jump(states, contacts, i) for i in range(states.n_objects)])


PLUGINS = (TrackJumpTruth,)
