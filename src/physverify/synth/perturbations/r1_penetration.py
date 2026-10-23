"""R1: a resting body B pushed horizontally into a resting neighbour A by a constant offset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from physverify.physics.geometry import signed_distance
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._frame_windows import (
    on_top_pairs,
    rest_mask,
    solid_at,
    supports_nothing,
)
from physverify.synth.perturbations._measure import (
    length_of,
    pair_penetration,
    penetration_between,
)
from physverify.synth.perturbations._state_edit import (
    PluginBase,
    Vec,
    pick,
    provisional_record,
    require_clear_shift,
    shift_records,
    translated,
)
from physverify.synth.placement import UP, solve_offset
from physverify.synth.settings import PerturbationCommon

NAME = "r1_penetration"
SOLVE_SLACK = 1e-6


class Params(PerturbationCommon):
    levels: tuple[float, float, float, float] = (0.05, 0.1, 0.2, 0.4)
    max_gap_m: float = 0.03
    min_clearance_m: float = 0.01


class Penetration(PluginBase[Params]):
    target: ClassVar[PerturbationTarget] = PerturbationTarget.TRAJECTORY
    name: ClassVar[str] = NAME
    target_checks: ClassVar[tuple[str, ...]] = ("R1",)
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
        offset = self._offset(states, a, b, magnitude)
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
            {"moved": str(ids[b]), "offset_m": offset.tolist()},
        )
        return moved, shift_records(contacts, str(ids[b]), offset), record

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        a, b = (states.object_index(o) for o in record.target_objects)
        return pair_penetration(states, a, b)

    def _offset(self, states: ObjectStateTable, a: int, b: int, magnitude: float) -> Vec:
        mid = states.n_frames // 2
        solid_a, solid_b = solid_at(states, a, mid), solid_at(states, b, mid)
        towards = states.position[mid, a] - states.position[mid, b]
        towards = towards - float(towards @ UP) * UP
        direction = towards / np.linalg.norm(towards)
        gap = max(0.0, signed_distance(solid_a, solid_b).distance)
        reach = gap + min(length_of(states, a), length_of(states, b))

        def per_frame(shift: float, frames: Sequence[int]) -> list[float]:
            moved = translated(states, b, 0, shift * direction)
            return [
                penetration_between(solid_at(states, a, t), solid_at(moved, b, t)) for t in frames
            ]

        # The max over frames is solved on a growing set of active frames: a subset
        # under-estimates it, so the frame that exceeds the target joins until none does.
        active = [mid]
        while True:
            shift = solve_offset(
                lambda s: max(per_frame(s, active)), magnitude, 0.0, reach, tol=1e-6
            )
            values = per_frame(shift, range(states.n_frames))
            worst = int(np.argmax(values))
            if worst in active or values[worst] <= magnitude + SOLVE_SLACK:
                break
            active.append(worst)
        out: Vec = shift * direction
        return out


def resting_pairs(
    states: ObjectStateTable, contacts: ContactTable, max_gap: float, tol: float
) -> list[tuple[int, int]]:
    """(A, B): both at rest the whole clip on the floor, clean gap <= ``max_gap``, nothing on B."""
    ids = states.object_ids.astype(str).tolist()
    on_floor = {up for up, low in on_top_pairs(contacts) if low == ENV_ID}
    still = [
        i for i, oid in enumerate(ids) if oid in on_floor and bool(np.all(rest_mask(states, i)))
    ]
    mid = states.n_frames // 2
    pairs: list[tuple[int, int]] = []
    for a in still:
        for b in still:
            if a == b or not supports_nothing(ids[b], contacts):
                continue
            gap = signed_distance(solid_at(states, a, mid), solid_at(states, b, mid)).distance
            if -tol <= gap <= max_gap:
                pairs.append((a, b))
    return pairs


PLUGINS = (Penetration,)
