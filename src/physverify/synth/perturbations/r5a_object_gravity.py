"""R5a: one ball follows g' = R(theta) g during its whole first flight while the other keeps g."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from physverify.physics.geometry import Solid
from physverify.schema.contacts import ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._ballistics import (
    SWITCH_DELAY_S,
    GravitySchedule,
    integrate_sphere,
    time_to_height,
)
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._edit import (
    PluginBase,
    before,
    bounce_record,
    centres_in_image,
    clearance,
    gravity_of,
    image_plane_tilt,
    keep_records,
    pick,
    provisional_record,
    radius_of,
    restitution_for_edit,
    spheres,
    with_flight,
)
from physverify.synth.perturbations._measure import R5A_MIN_FRAMES, flight_fits, inconsistency
from physverify.synth.perturbations._windows import solid_at
from physverify.synth.settings import PerturbationCommon

NAME = "r5a_object_gravity"
# The tilted gravity ends a hair after the first landing so the bounce itself is solved with g'.


class Params(PerturbationCommon):
    levels: tuple[float, float, float, float] = (0.05, 0.1, 0.2, 0.4)


class ObjectGravity(PluginBase[Params]):
    target: ClassVar[PerturbationTarget] = PerturbationTarget.TRAJECTORY
    name: ClassVar[str] = NAME
    target_checks: ClassVar[tuple[str, ...]] = ("R5a",)
    magnitude_unit: ClassVar[str] = "accel_rel_diff"
    magnitude_kind: ClassVar[MagnitudeKind] = MagnitudeKind.CHECK_UNIT
    Params: ClassVar[type[PerturbationCommon]] = Params
    families: ClassVar[tuple[str, ...]] = ("f1_projectile",)

    def apply(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        magnitude: float,
        rng: np.random.Generator,
    ) -> tuple[ObjectStateTable, ContactTable, PerturbationRecord]:
        balls = spheres(states)
        if len(balls) < 2:
            raise PerturbationNotApplicableError(f"{NAME}: needs two balls")
        index = int(pick(rng, balls))
        reference = int(pick(rng, [b for b in balls if b != index]))
        sign = float(pick(rng, (-1.0, 1.0)))
        g = gravity_of(scene)
        tilted = image_plane_tilt(scene, magnitude, sign) @ g
        radius = radius_of(states, index)
        p0, v0, t0 = states.position[0, index], states.velocity[0, index], float(states.t[0])
        landing = t0 + time_to_height(float(p0[2]), float(v0[2]), float(tilted[2]), radius)
        if not np.isfinite(landing):
            raise PerturbationNotApplicableError(f"{NAME}: the ball never lands under g'")
        schedule = GravitySchedule((landing + SWITCH_DELAY_S,), (tilted, g))
        e_clean = restitution_for_edit(scene, states, contacts, index)
        flight = integrate_sphere(p0, v0, t0, radius, schedule, e_clean, states.t)
        moved = with_flight(states, index, 0, flight)
        self._check(scene, moved, index)
        object_id = str(states.object_ids[index])
        fps = float(scene.frame_rate)
        records = [bounce_record(object_id, b, radius, fps) for b in flight.bounces]
        new_contacts = keep_records(contacts, before([object_id], t0 * fps), records)
        record = provisional_record(
            self,
            magnitude,
            (object_id, str(states.object_ids[reference])),
            (0, states.n_frames - 1),
            {
                "gravity": tilted.tolist(),
                "clean_gravity": g.tolist(),
                "tilt_sign": sign,
                "e_clean": e_clean,
            },
        )
        return moved, new_contacts, record

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        """R5a between the first flights of the tilted ball and the reference ball."""
        fits = []
        for object_id in record.target_objects:
            flights = flight_fits(states, contacts, states.object_index(object_id), R5A_MIN_FRAMES)
            if not flights:
                return float("nan")
            fits.append(flights[0])
        return inconsistency(fits)

    @staticmethod
    def _check(scene: SceneSpec, moved: ObjectStateTable, index: int) -> None:
        def body(t: int) -> Solid:
            return solid_at(moved, index, t)

        if not centres_in_image(scene, moved.position[:, index]):
            raise PerturbationNotApplicableError(f"{NAME}: the tilted flight leaves the image")
        if clearance(moved, body, index, range(moved.n_frames)) <= 0.0:
            raise PerturbationNotApplicableError(f"{NAME}: the tilted flight hits the other ball")


PLUGINS = (ObjectGravity,)
