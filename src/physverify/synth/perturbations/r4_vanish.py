"""R4 (missing): a body that supports nothing is hidden for n = m consecutive frames while it is
inside the image and its projected silhouette overlaps no other one."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._frame_windows import (
    Mask,
    in_image_fraction,
    polygons_overlap,
    projected_silhouette,
    segments,
    solid_at,
    supports_nothing,
)
from physverify.synth.perturbations._measure import object_missing
from physverify.synth.perturbations._state_edit import (
    PluginBase,
    camera_of,
    edit_object,
    pick,
    provisional_record,
)
from physverify.synth.settings import PerturbationCommon

NAME = "r4_vanish"


class Params(PerturbationCommon):
    levels: tuple[float, float, float, float] = (1.0, 2.0, 4.0, 8.0)
    # Share of the projected silhouette that must fall inside the image in every hidden frame.
    min_in_image: float = 0.99


class Vanish(PluginBase[Params]):
    target: ClassVar[PerturbationTarget] = PerturbationTarget.TRAJECTORY
    name: ClassVar[str] = NAME
    target_checks: ClassVar[tuple[str, ...]] = ("R4",)
    magnitude_unit: ClassVar[str] = "frames"
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
        length = round(magnitude)
        ids = states.object_ids.astype(str).tolist()
        options = [
            (i, lo, hi)
            for i, object_id in enumerate(ids)
            if supports_nothing(object_id, contacts)
            for lo, hi in segments(self._isolated(scene, states, i), length)
        ]
        if not options:
            raise PerturbationNotApplicableError(
                f"{NAME}: no body is isolated in the image for {length} frames"
            )
        index, lo, hi = pick(rng, options)
        first = int(rng.integers(lo, hi - length + 1))
        window = slice(first, first + length)
        hidden = edit_object(states, index, window, visible=False)
        record = provisional_record(
            self,
            float(length),
            (ids[index],),
            (first, first + length - 1),
            {"window": [first, first + length - 1]},
        )
        return hidden, contacts, record

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        return object_missing(states, states.object_index(record.target_objects[0]))

    def _isolated(self, scene: SceneSpec, states: ObjectStateTable, index: int) -> Mask:
        """Frames where the body's silhouette is inside the image and apart from every other one."""
        K, pose = camera_of(scene)  # noqa: N806
        mask = np.zeros(states.n_frames, dtype=bool)
        for t in range(states.n_frames):
            polygons = [
                projected_silhouette(K, pose, solid_at(states, j, t))
                for j in range(states.n_objects)
            ]
            own = polygons[index]
            inside = in_image_fraction(own, scene.resolution) >= self.params.min_in_image
            apart = not any(polygons_overlap(own, p) for j, p in enumerate(polygons) if j != index)
            mask[t] = inside and apart
        return mask


PLUGINS = (Vanish,)
