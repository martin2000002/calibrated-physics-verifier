"""R1 decoy: the body ``r1_penetration`` would push into its neighbour, translated the same distance
the opposite way. Same edit footprint, no crossing: a clean clip for training labels."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.magnitudes import pair_penetration
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._r1_pairs import penetration_offset, resting_pairs
from physverify.synth.perturbations._state_edit import (
    PluginBase,
    pick,
    provisional_record,
    require_clear_shift,
    shift_records,
    translated,
)
from physverify.synth.settings import PerturbationCommon

NAME = "r1_penetration_decoy"
MIRRORS = "r1_penetration"


class Params(PerturbationCommon):
    levels: tuple[float, float, float, float] = (0.05, 0.1, 0.2, 0.4)
    max_gap_m: float = 0.03
    min_clearance_m: float = 0.01


class PenetrationDecoy(PluginBase[Params]):
    target: ClassVar[PerturbationTarget] = PerturbationTarget.TRAJECTORY
    name: ClassVar[str] = NAME
    seed_stream: ClassVar[str] = MIRRORS
    target_checks: ClassVar[tuple[str, ...]] = ()
    magnitude_unit: ClassVar[str] = "pen_over_size"
    magnitude_kind: ClassVar[MagnitudeKind] = MagnitudeKind.CHECK_UNIT
    Params: ClassVar[type[PerturbationCommon]] = Params
    families: ClassVar[tuple[str, ...]] = ("f5_contact",)

    def apply(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        magnitude: float,
        rng: np.random.Generator,
    ) -> tuple[ObjectStateTable, ContactTable, PerturbationRecord]:
        pairs = resting_pairs(states, contacts, self.params.max_gap_m, self.params.contact_tol_m)
        if not pairs:
            raise PerturbationNotApplicableError(
                f"{NAME}: no resting pair with a clean gap <= {self.params.max_gap_m} m"
            )
        a, b = pick(rng, pairs)
        offset = -penetration_offset(states, a, b, magnitude)
        moved = translated(states, b, 0, offset)
        require_clear_shift(
            scene,
            moved,
            b,
            a,
            self.params.min_clearance_m,
            too_close=f"{NAME}: the shifted body comes too close to a third body",
            leaves=f"{NAME}: the shifted body leaves the image",
        )
        ids = states.object_ids.astype(str)
        record = provisional_record(
            self,
            magnitude,
            (str(ids[a]), str(ids[b])),
            (0, states.n_frames - 1),
            {"moved": str(ids[b]), "offset_m": offset.tolist(), "mirrors": MIRRORS},
        )
        return moved, shift_records(contacts, str(ids[b]), offset), record

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        a, b = (states.object_index(o) for o in record.target_objects)
        return pair_penetration(states, a, b)


PLUGINS = (PenetrationDecoy,)
