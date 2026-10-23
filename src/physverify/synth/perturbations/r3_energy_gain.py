"""R3b: the post-collision velocities of a ball pair rescaled about the centre of mass so the
kinetic energy grows by the fraction m while momentum is conserved (ADR 0004 §3.5)."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.magnitudes import collision_velocities, pair_energy_gain
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._ballistics import (
    GravitySchedule,
    energy_scale,
    integrate_sphere,
)
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._state_edit import (
    PluginBase,
    Vec,
    ballistic_state,
    before,
    bounce_record,
    gravity_of,
    keep_records,
    object_mass,
    provisional_record,
    radius_of,
    require_in_image,
    restitution_for_edit,
    with_flight,
)
from physverify.synth.settings import PerturbationCommon

NAME = "r3_energy_gain"


class Params(PerturbationCommon):
    levels: tuple[float, float, float, float] = (0.1, 0.25, 0.5, 1.0)


class EnergyGain(PluginBase[Params]):
    target: ClassVar[PerturbationTarget] = PerturbationTarget.TRAJECTORY
    name: ClassVar[str] = NAME
    target_checks: ClassVar[tuple[str, ...]] = ("R3b",)
    magnitude_unit: ClassVar[str] = "dE_over_E"
    magnitude_kind: ClassVar[MagnitudeKind] = MagnitudeKind.CHECK_UNIT
    Params: ClassVar[type[PerturbationCommon]] = Params
    families: ClassVar[tuple[str, ...]] = ("f3_collision",)

    def apply(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        magnitude: float,
        rng: np.random.Generator,
    ) -> tuple[ObjectStateTable, ContactTable, PerturbationRecord]:
        impact = first_ball_collision(states, contacts)
        if impact is None:
            raise PerturbationNotApplicableError(f"{NAME}: no collision between two balls")
        fps = float(scene.frame_rate)
        g = gravity_of(scene)
        t_c = impact.t_start / fps
        pair = (states.object_index(impact.object_a), states.object_index(impact.object_b))
        masses = [object_mass(scene, impact.object_a), object_mass(scene, impact.object_b)]
        tol = self.params.contact_tol_m
        velocities = [collision_velocities(states, contacts, i, t_c, g, tol) for i in pair]
        v_minus, v_plus = [v[0] for v in velocities], [v[1] for v in velocities]
        scale = energy_scale(masses, v_minus, v_plus, magnitude)
        if not math.isfinite(scale):
            raise PerturbationNotApplicableError(f"{NAME}: the energy gain cannot be reached")
        m = np.asarray(masses)
        cm_minus, cm_plus = m @ np.array(v_minus) / m.sum(), m @ np.array(v_plus) / m.sum()
        start = int(np.searchsorted(states.t, t_c, side="right"))
        moved, records = states, []
        for index, after in zip(pair, v_plus, strict=True):
            moved, bounces = self._reflight(
                scene,
                moved,
                states,
                contacts,
                index,
                t_c,
                cm_minus + scale * (after - cm_plus),
                start,
            )
            records.extend(bounces)
        self._check(scene, moved, pair, start)
        ids = (impact.object_a, impact.object_b)
        new_contacts = keep_records(contacts, before(ids, impact.t_start + 1e-9), records)
        record = provisional_record(
            self,
            magnitude,
            ids,
            (max(0, start - 1), states.n_frames - 1),
            {
                "t_c": t_c,
                "mass_ratio": masses[0] / masses[1],
                "scale": scale,
                "gravity": g.tolist(),
            },
        )
        return moved, new_contacts, record

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        pair = (
            states.object_index(record.target_objects[0]),
            states.object_index(record.target_objects[1]),
        )
        g = np.asarray(record.params["gravity"], dtype=np.float64)
        ratio, t_c = float(record.params["mass_ratio"]), float(record.params["t_c"])
        return pair_energy_gain(states, contacts, pair, ratio, t_c, g, self.params.contact_tol_m)

    @staticmethod
    def _reflight(
        scene: SceneSpec,
        moved: ObjectStateTable,
        clean: ObjectStateTable,
        contacts: ContactTable,
        index: int,
        t_c: float,
        velocity: Vec,
        start: int,
    ) -> tuple[ObjectStateTable, list[ContactRecord]]:
        g = gravity_of(scene)
        radius = radius_of(clean, index)
        p_c, _ = ballistic_state(clean, index, t_c, g)
        e_clean = restitution_for_edit(scene, clean, contacts, index)
        flight = integrate_sphere(
            p_c, velocity, t_c, radius, GravitySchedule.constant(g), e_clean, clean.t[start:]
        )
        object_id = str(clean.object_ids[index])
        fps = float(scene.frame_rate)
        return with_flight(moved, index, start, flight), [
            bounce_record(object_id, b, radius, fps) for b in flight.bounces
        ]

    @staticmethod
    def _check(
        scene: SceneSpec, moved: ObjectStateTable, pair: tuple[int, int], start: int
    ) -> None:
        a, b = pair
        gaps = np.linalg.norm(moved.position[start:, a] - moved.position[start:, b], axis=1)
        if np.any(gaps <= radius_of(moved, a) + radius_of(moved, b)):
            raise PerturbationNotApplicableError(f"{NAME}: the balls would touch again")
        require_in_image(scene, moved.position[start:, [a, b]], f"{NAME}: a ball leaves the image")


def first_ball_collision(states: ObjectStateTable, contacts: ContactTable) -> ContactRecord | None:
    def is_ball(object_id: str) -> bool:
        return str(states.primitive[states.object_index(object_id)]) == PrimitiveKind.SPHERE.value

    impacts = [
        r
        for r in contacts.records
        if r.kind is ContactKind.IMPACT
        and ENV_ID not in (r.object_a, r.object_b)
        and is_ball(r.object_a)
        and is_ball(r.object_b)
    ]
    return min(impacts, key=lambda r: r.t_start) if impacts else None


PLUGINS = (EnergyGain,)
