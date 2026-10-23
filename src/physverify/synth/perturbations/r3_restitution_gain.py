"""R3a: the first floor bounce of a ball re-integrated with restitution e' = 1 + m."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from physverify.schema.contacts import ContactTable
from physverify.schema.scene import MagnitudeKind, PerturbationRecord, SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.ballistics import Bounce, GravitySchedule, integrate_sphere
from physverify.synth.magnitudes import bounce_restitution, floor_bounce
from physverify.synth.perturb import PerturbationNotApplicableError
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.perturbations._state_edit import (
    PluginBase,
    ballistic_state,
    before,
    bounce_record,
    first_floor_impact,
    gravity_of,
    keep_records,
    pick,
    provisional_record,
    radius_of,
    require_clear_flight,
    restitution_for_edit,
    spheres,
    with_flight,
)
from physverify.synth.settings import PerturbationCommon

NAME = "r3_restitution_gain"


class Params(PerturbationCommon):
    levels: tuple[float, float, float, float] = (0.05, 0.1, 0.25, 0.5)


class RestitutionGain(PluginBase[Params]):
    target: ClassVar[PerturbationTarget] = PerturbationTarget.TRAJECTORY
    name: ClassVar[str] = NAME
    target_checks: ClassVar[tuple[str, ...]] = ("R3a",)
    magnitude_unit: ClassVar[str] = "e_minus_1"
    magnitude_kind: ClassVar[MagnitudeKind] = MagnitudeKind.CHECK_UNIT
    Params: ClassVar[type[PerturbationCommon]] = Params
    families: ClassVar[tuple[str, ...]] = ("f2_bounce",)

    def apply(
        self,
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        magnitude: float,
        rng: np.random.Generator,
    ) -> tuple[ObjectStateTable, ContactTable, PerturbationRecord]:
        ids = states.object_ids.astype(str).tolist()
        balls = [i for i in spheres(states) if first_floor_impact(ids[i], contacts) is not None]
        if not balls:
            raise PerturbationNotApplicableError(f"{NAME}: no ball bounces on the floor")
        index = pick(rng, balls)
        object_id = ids[index]
        impact = first_floor_impact(object_id, contacts)
        g = gravity_of(scene)
        bounce = None if impact is None else floor_bounce(states, index, impact.t_start, g)
        if impact is None or bounce is None:
            raise PerturbationNotApplicableError(
                f"{NAME}: the first floor bounce cannot be resolved"
            )
        t_c, v_minus, _ = bounce
        radius = radius_of(states, index)
        p_c, _ = ballistic_state(states, index, t_c, g)
        p_c[2] = radius
        v_plus = np.array([v_minus[0], v_minus[1], -(1.0 + magnitude) * v_minus[2]])
        e_clean = restitution_for_edit(scene, states, contacts, index)
        start = int(np.searchsorted(states.t, t_c, side="right"))
        flight = integrate_sphere(
            p_c, v_plus, t_c, radius, GravitySchedule.constant(g), e_clean, states.t[start:]
        )
        moved = with_flight(states, index, start, flight)
        apex = p_c + v_plus * (v_plus[2] / -g[2]) + 0.5 * g * (v_plus[2] / -g[2]) ** 2
        require_clear_flight(
            scene,
            moved,
            index,
            range(start, moved.n_frames),
            np.vstack([moved.position[start:, index], apex]),
            leaves=f"{NAME}: the new rebound leaves the image",
            hits=f"{NAME}: the new rebound hits another body",
        )
        fps = float(scene.frame_rate)
        first = Bounce(t_c, p_c, v_minus, v_plus)
        records = [bounce_record(object_id, b, radius, fps) for b in (first, *flight.bounces)]
        new_contacts = keep_records(contacts, before([object_id], impact.t_start), records)
        record = provisional_record(
            self,
            magnitude,
            (object_id,),
            (max(0, start - 1), states.n_frames - 1),
            {
                "impact_frame": t_c * fps,
                "restitution": 1.0 + magnitude,
                "e_clean": e_clean,
                "gravity": g.tolist(),
            },
        )
        return moved, new_contacts, record

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        index = states.object_index(record.target_objects[0])
        g = np.asarray(record.params["gravity"], dtype=np.float64)
        return bounce_restitution(states, index, float(record.params["impact_frame"]), g) - 1.0


PLUGINS = (RestitutionGain,)
