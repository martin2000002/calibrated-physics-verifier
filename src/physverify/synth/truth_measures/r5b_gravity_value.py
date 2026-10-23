"""R5b truth: ``gravity_scale_error`` over the sphere flights."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.magnitudes import gravity_scale_error


class GravityScaleTruth:
    check_id: ClassVar[str] = "R5b"

    def measure(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        tol: float,
        cuts: Sequence[int],
    ) -> float:
        return gravity_scale_error(states, contacts, cuts)


PLUGINS = (GravityScaleTruth,)
