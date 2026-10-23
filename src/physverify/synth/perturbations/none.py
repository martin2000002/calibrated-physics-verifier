"""Identity perturbation: the clean clip, level 0 and magnitude 0."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._state_edit import PluginBase, provisional_record
from physverify.synth.settings import PerturbationCommon

NAME = "none"


class Params(PerturbationCommon):
    levels: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


class NoPerturbation(PluginBase[Params]):
    target: ClassVar[PerturbationTarget] = PerturbationTarget.TRAJECTORY
    name: ClassVar[str] = NAME
    target_checks: ClassVar[tuple[str, ...]] = ()
    magnitude_unit: ClassVar[str] = "none"
    magnitude_kind: ClassVar[MagnitudeKind] = MagnitudeKind.CHECK_UNIT
    Params: ClassVar[type[PerturbationCommon]] = Params

    def apply(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        magnitude: float,
        rng: np.random.Generator,
    ) -> tuple[ObjectStateTable, ContactTable, PerturbationRecord]:
        return states, contacts, provisional_record(self, 0.0, (), None, {})

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        return 0.0


PLUGINS = (NoPerturbation,)
