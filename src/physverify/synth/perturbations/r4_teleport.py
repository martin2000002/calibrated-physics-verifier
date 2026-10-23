"""R4 (jump): a resting body standing only on the floor is shifted horizontally by D = m L from a
frame f in [T/4, 3T/4] to the end of the clip."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._edit import (
    PluginBase,
    pick,
    provisional_record,
    require_clearance,
    require_in_image,
    translated,
)
from physverify.synth.perturbations._measure import length_of, object_jump
from physverify.synth.perturbations._windows import (
    on_top_pairs,
    rest_mask,
    supports_nothing,
)
from physverify.synth.scene_kit import heading
from physverify.synth.settings import PerturbationCommon

NAME = "r4_teleport"


class Params(PerturbationCommon):
    levels: tuple[float, float, float, float] = (0.25, 0.5, 1.0, 2.0)
    min_clearance_m: float = 0.02
    start_range: tuple[float, float] = (0.25, 0.75)


class Teleport(PluginBase[Params]):
    target: ClassVar[PerturbationTarget] = PerturbationTarget.TRAJECTORY
    name: ClassVar[str] = NAME
    target_checks: ClassVar[tuple[str, ...]] = ("R4",)
    magnitude_unit: ClassVar[str] = "jump_over_size"
    magnitude_kind: ClassVar[MagnitudeKind] = MagnitudeKind.CHECK_UNIT
    Params: ClassVar[type[PerturbationCommon]] = Params
    families: ClassVar[tuple[str, ...]] = ("f2_bounce", "f4_stack", "f5_contact")

    def apply(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        magnitude: float,
        rng: np.random.Generator,
    ) -> tuple[ObjectStateTable, ContactTable, PerturbationRecord]:
        candidates = floor_resting(states, contacts)
        if not candidates:
            raise PerturbationNotApplicableError(
                f"{NAME}: no body rests on the floor alone with nothing on top"
            )
        index = pick(rng, candidates)
        n = states.n_frames
        lo = max(1, math.ceil(n * self.params.start_range[0]))
        hi = min(n - 2, max(lo, math.floor(n * self.params.start_range[1])))
        first = int(rng.integers(lo, hi + 1))
        offset = (
            magnitude * length_of(states, index) * heading(float(rng.uniform(0.0, 2.0 * math.pi)))
        )
        moved = translated(states, index, first, offset)
        require_clearance(
            moved,
            index,
            range(first, n),
            self.params.min_clearance_m,
            f"{NAME}: the jump lands too close to another body",
        )
        require_in_image(scene, moved.position[first:, index], f"{NAME}: the jump leaves the image")
        object_id = str(states.object_ids[index])
        record = provisional_record(
            self,
            magnitude,
            (object_id,),
            (first - 1, first),
            {"frame": first, "offset_m": offset.tolist()},
        )
        return moved, contacts, record

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        return object_jump(states, contacts, states.object_index(record.target_objects[0]))


def floor_resting(states: ObjectStateTable, contacts: ContactTable) -> list[int]:
    """Bodies at rest the whole clip, supported by the floor alone, with nothing on top."""
    pairs = on_top_pairs(contacts)
    ids = states.object_ids.astype(str).tolist()
    return [
        i
        for i, object_id in enumerate(ids)
        if {low for up, low in pairs if up == object_id} == {ENV_ID}
        and supports_nothing(object_id, contacts)
        and bool(np.all(rest_mask(states, i)))
    ]


PLUGINS = (Teleport,)
