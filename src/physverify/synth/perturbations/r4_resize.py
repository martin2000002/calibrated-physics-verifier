"""R4 (size): a flying ball rendered (1 + m) times larger for a window of w < T/2 frames, so the
median size is unchanged and ``size_change_ratio`` equals m."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from physverify.physics.measures.object_integrity import size_change_ratio
from physverify.schema.contacts import ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._frame_windows import flight_mask, rest_mask, segments
from physverify.synth.perturbations._state_edit import (
    PluginBase,
    edit_object,
    pick,
    provisional_record,
    require_clearance,
    spheres,
)
from physverify.synth.settings import PerturbationCommon

NAME = "r4_resize"


class Params(PerturbationCommon):
    levels: tuple[float, float, float, float] = (0.05, 0.1, 0.2, 0.4)
    window_range: tuple[int, int] = (3, 8)


class Resize(PluginBase[Params]):
    target: ClassVar[PerturbationTarget] = PerturbationTarget.TRAJECTORY
    name: ClassVar[str] = NAME
    target_checks: ClassVar[tuple[str, ...]] = ("R4",)
    magnitude_unit: ClassVar[str] = "rel_size_change"
    magnitude_kind: ClassVar[MagnitudeKind] = MagnitudeKind.CHECK_UNIT
    Params: ClassVar[type[PerturbationCommon]] = Params
    families: ClassVar[tuple[str, ...]] = ("f1_projectile", "f2_bounce")

    def apply(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        magnitude: float,
        rng: np.random.Generator,
    ) -> tuple[ObjectStateTable, ContactTable, PerturbationRecord]:
        shortest, longest = self.params.window_range
        longest = min(longest, (states.n_frames - 1) // 2)
        if longest < shortest:
            raise PerturbationNotApplicableError(
                f"{NAME}: the clip is too short for a window of {shortest} frames"
            )
        width = int(rng.integers(shortest, longest + 1))
        ids = states.object_ids.astype(str).tolist()
        options = [
            (i, lo, hi)
            for i in spheres(states)
            for lo, hi in segments(
                flight_mask(ids[i], contacts, states.n_frames) & ~rest_mask(states, i), width
            )
        ]
        if not options:
            raise PerturbationNotApplicableError(f"{NAME}: no ball flies for {width} frames")
        index, lo, hi = pick(rng, options)
        first = int(rng.integers(lo, hi - width + 1))
        window = slice(first, first + width)
        resized = edit_object(
            states, index, window, size=states.size[window, index] * (1.0 + magnitude)
        )
        require_clearance(
            resized,
            index,
            range(first, first + width),
            self.params.contact_tol_m,
            f"{NAME}: the enlarged ball would touch another body",
            with_floor=True,
        )
        record = provisional_record(
            self,
            magnitude,
            (ids[index],),
            (first, first + width - 1),
            {"window": [first, first + width - 1], "scale": 1.0 + magnitude},
        )
        return resized, contacts, record

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        return size_change_ratio(states.size[:, states.object_index(record.target_objects[0])])


PLUGINS = (Resize,)
