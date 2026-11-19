"""R3b: the post-collision velocities of a ball pair rescaled about the centre of mass so the
kinetic energy grows by the fraction m while momentum is conserved."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.ballistics import (
    GravitySchedule,
    energy_scale,
)
from physverify.synth.magnitudes import collision_velocities, pair_energy_gain
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._reflight import Launch, reflight
from physverify.synth.perturbations._state_edit import (
    PluginBase,
    Vec,
    ballistic_state,
    before,
    gravity_of,
    keep_records,
    object_mass,
    provisional_record,
    radius_of,
    require_in_image,
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
        collision = _collision(scene, states, contacts, self.params.contact_tol_m)
        t_c, masses = collision.t_c, collision.masses
        scale = energy_scale(masses, collision.v_minus, collision.v_plus, magnitude)
        if not math.isfinite(scale):
            raise PerturbationNotApplicableError(f"{NAME}: the energy gain cannot be reached")
        start = int(np.searchsorted(states.t, t_c, side="right"))
        moved, records = self._reflown(scene, states, contacts, collision, scale, start)
        self._check(scene, moved, collision.pair, start)
        ids = (collision.impact.object_a, collision.impact.object_b)
        kept = before(ids, collision.impact.t_start + 1e-9)
        new_contacts = keep_records(contacts, kept, records)
        record = provisional_record(
            self,
            magnitude,
            ids,
            (max(0, start - 1), states.n_frames - 1),
            {
                "t_c": t_c,
                "mass_ratio": masses[0] / masses[1],
                "scale": scale,
                "gravity": gravity_of(scene).tolist(),
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
    def _reflown(
        scene: SceneSpec,
        clean: ObjectStateTable,
        contacts: ContactTable,
        collision: _Collision,
        scale: float,
        start: int,
    ) -> tuple[ObjectStateTable, list[ContactRecord]]:
        """Both balls re-flown from t_c with the post-collision velocities rescaled about the
        centre of mass by ``scale``."""
        g = gravity_of(scene)
        m = np.asarray(collision.masses)
        cm_minus = m @ np.array(collision.v_minus) / m.sum()
        cm_plus = m @ np.array(collision.v_plus) / m.sum()
        moved, records = clean, []
        for index, after in zip(collision.pair, collision.v_plus, strict=True):
            p_c, _ = ballistic_state(clean, index, collision.t_c, g)
            velocity = cm_minus + scale * (after - cm_plus)
            launch = Launch(p_c, velocity, collision.t_c, GravitySchedule.constant(g))
            reflown = reflight(scene, clean, contacts, index, start, launch, target=moved)
            moved = reflown.states
            records.extend(reflown.records)
        return moved, records

    @staticmethod
    def _check(
        scene: SceneSpec, moved: ObjectStateTable, pair: tuple[int, int], start: int
    ) -> None:
        a, b = pair
        gaps = np.linalg.norm(moved.position[start:, a] - moved.position[start:, b], axis=1)
        if np.any(gaps <= radius_of(moved, a) + radius_of(moved, b)):
            raise PerturbationNotApplicableError(f"{NAME}: the balls would touch again")
        require_in_image(scene, moved.position[start:, [a, b]], f"{NAME}: a ball leaves the image")


@dataclass(frozen=True)
class _Collision:
    """The first ball-ball impact, its time t_c (s), and each ball's mass and velocities."""

    impact: ContactRecord
    t_c: float
    pair: tuple[int, int]
    masses: list[float]
    v_minus: list[Vec]
    v_plus: list[Vec]


def _collision(
    scene: SceneSpec, states: ObjectStateTable, contacts: ContactTable, tol: float
) -> _Collision:
    impact = first_ball_collision(states, contacts)
    if impact is None:
        raise PerturbationNotApplicableError(f"{NAME}: no collision between two balls")
    g = gravity_of(scene)
    t_c = impact.t_start / float(scene.frame_rate)
    pair = (states.object_index(impact.object_a), states.object_index(impact.object_b))
    masses = [object_mass(scene, impact.object_a), object_mass(scene, impact.object_b)]
    velocities = [collision_velocities(states, contacts, i, t_c, g, tol) for i in pair]
    return _Collision(
        impact, t_c, pair, masses, [v[0] for v in velocities], [v[1] for v in velocities]
    )


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
