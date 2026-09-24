"""R5b: the scene is re-simulated under k g with k = 1 + sigma m (sigma = +-1)."""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Literal

import numpy as np

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._edit import (
    PluginBase,
    centres_in_image,
    gravity_of,
    pick,
    provisional_record,
)
from physverify.synth.perturbations._measure import gravity_scale_error
from physverify.synth.scene_kit import sphere_path
from physverify.synth.settings import PerturbationCommon

NAME = "r5b_gravity_scale"
SIGNS: dict[str, tuple[float, ...]] = {"both": (-1.0, 1.0), "up": (1.0,), "down": (-1.0,)}


class Params(PerturbationCommon):
    levels: tuple[float, float, float, float] = (0.05, 0.1, 0.2, 0.4)
    direction: Literal["both", "up", "down"] = "both"


class GravityScale(PluginBase[Params]):
    target: ClassVar[PerturbationTarget] = PerturbationTarget.SCENE
    name: ClassVar[str] = NAME
    target_checks: ClassVar[tuple[str, ...]] = ("R5b",)
    magnitude_unit: ClassVar[str] = "g_rel_error"
    magnitude_kind: ClassVar[MagnitudeKind] = MagnitudeKind.CHECK_UNIT
    Params: ClassVar[type[PerturbationCommon]] = Params
    families: ClassVar[tuple[str, ...]] = ("f1_projectile", "f2_bounce")

    def apply(
        self, scene: SceneSpec, magnitude: float, rng: np.random.Generator
    ) -> tuple[SceneSpec, PerturbationRecord]:
        scale = 1.0 + float(pick(rng, SIGNS[self.params.direction])) * magnitude
        if scale <= 0.0:
            raise PerturbationNotApplicableError(f"{NAME}: k = {scale:g} is not a gravity")
        gravity = scale * gravity_of(scene)
        edited = dataclasses.replace(
            scene, gravity=(float(gravity[0]), float(gravity[1]), float(gravity[2]))
        )
        balls = [o for o in scene.objects if o.primitive == PrimitiveKind.SPHERE]
        if not balls:
            raise PerturbationNotApplicableError(f"{NAME}: the scene has no ball")
        times = np.arange(scene.settle_frames + scene.n_frames, dtype=np.float64) / scene.frame_rate
        paths = [
            sphere_path(o.position, o.velocity, gravity, o.extents[0] / 2.0, times) for o in balls
        ]
        if not centres_in_image(edited, np.vstack(paths)):
            raise PerturbationNotApplicableError(f"{NAME}: a ball leaves the image under k g")
        record = provisional_record(
            self,
            magnitude,
            tuple(o.object_id for o in balls),
            None,
            {"gravity_scale": scale, "gravity": gravity.tolist()},
        )
        return edited, record

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        return gravity_scale_error(states, contacts)


PLUGINS = (GravityScale,)
