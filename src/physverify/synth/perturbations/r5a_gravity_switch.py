"""R5a: from a switch instant t_s every ball in flight follows g' = R(theta) g until it lands,
then g again. The measure compares window fits on each side of t_s."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

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
from physverify.synth.perturbations._frame_windows import flight_mask, rest_mask, segments
from physverify.synth.perturbations._measure import (
    SIDE_FRAMES,
    SWITCH_FRAME,
    fit_frames,
    inconsistency,
)
from physverify.synth.perturbations._state_edit import (
    PluginBase,
    Vec,
    ballistic_state,
    before,
    bounce_record,
    gravity_of,
    image_plane_tilt,
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

NAME = "r5a_gravity_switch"


class Params(PerturbationCommon):
    levels: tuple[float, float, float, float] = (0.05, 0.1, 0.2, 0.4)


def flight_window(
    states: ObjectStateTable, contacts: ContactTable, index: int, frame: int
) -> tuple[int, int] | None:
    """Half-open flight segment of object ``index`` that contains ``frame``."""
    mask = flight_mask(str(states.object_ids[index]), contacts, states.n_frames) & ~rest_mask(
        states, index
    )
    return next(((lo, hi) for lo, hi in segments(mask, 1) if lo <= frame < hi), None)


def switch_frames(states: ObjectStateTable, contacts: ContactTable, index: int) -> list[int]:
    """Frames s whose first-flight windows [lo, s) and [s, hi) both have SIDE_FRAMES frames."""
    mask = flight_mask(str(states.object_ids[index]), contacts, states.n_frames) & ~rest_mask(
        states, index
    )
    found = segments(mask, 2 * SIDE_FRAMES)
    if not found:
        return []
    lo, hi = found[0]
    return list(range(lo + SIDE_FRAMES, hi - SIDE_FRAMES + 1))


def in_flight(states: ObjectStateTable, contacts: ContactTable, index: int, frame: int) -> bool:
    window = flight_window(states, contacts, index, frame)
    return window is not None and window[0] <= frame - 1


class GravitySwitch(PluginBase[Params]):
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
        choices = [(i, s) for i in spheres(states) for s in switch_frames(states, contacts, i)]
        if not choices:
            raise PerturbationNotApplicableError(
                f"{NAME}: no ball flies {SIDE_FRAMES} frames on each side"
            )
        main, frame = pick(rng, choices)
        sign = float(pick(rng, (-1.0, 1.0)))
        g = gravity_of(scene)
        tilted = image_plane_tilt(scene, magnitude, sign) @ g
        t_switch = float(states.t[frame]) - 0.5 / float(scene.frame_rate)
        others = [i for i in spheres(states) if i != main and in_flight(states, contacts, i, frame)]
        moved, records = states, []
        for index in (main, *others):
            moved, bounces = self._switched(
                scene, moved, contacts, index, frame, t_switch, (tilted, g)
            )
            records.extend(bounces)
        ids = [str(states.object_ids[i]) for i in (main, *others)]
        for index in (main, *others):
            require_clear_flight(
                scene,
                moved,
                index,
                range(frame, moved.n_frames),
                moved.position[frame:, index],
                leaves=f"{NAME}: a switched flight leaves the image",
                hits=f"{NAME}: a switched flight hits another ball",
            )
        new_contacts = keep_records(contacts, before(ids, t_switch * scene.frame_rate), records)
        record = provisional_record(
            self,
            magnitude,
            tuple(ids),
            (frame - 1, states.n_frames - 1),
            {
                SWITCH_FRAME: frame,
                "gravity": tilted.tolist(),
                "clean_gravity": g.tolist(),
                "tilt_sign": sign,
            },
        )
        return moved, new_contacts, record

    def measure(
        self, states: ObjectStateTable, contacts: ContactTable, record: PerturbationRecord
    ) -> float:
        """R5a between the fits of the main ball on [lo, s) and [s, hi)."""
        frame = int(record.params[SWITCH_FRAME])
        index = states.object_index(record.target_objects[0])
        window = flight_window(states, contacts, index, frame)
        if window is None or frame - window[0] < SIDE_FRAMES or window[1] - frame < SIDE_FRAMES:
            return math.nan
        lo, hi = window
        fits = [fit_frames(states, index, np.arange(a, b)) for a, b in ((lo, frame), (frame, hi))]
        return inconsistency(fits)

    @staticmethod
    def _switched(
        scene: SceneSpec,
        states: ObjectStateTable,
        contacts: ContactTable,
        index: int,
        frame: int,
        t_switch: float,
        gravities: tuple[Vec, Vec],
    ) -> tuple[ObjectStateTable, list]:
        tilted, g = gravities
        radius = radius_of(states, index)
        p, v = ballistic_state(states, index, t_switch, g)
        landing = t_switch + time_to_height(float(p[2]), float(v[2]), float(tilted[2]), radius)
        if not math.isfinite(landing):
            raise PerturbationNotApplicableError(f"{NAME}: a ball never lands under g'")
        schedule = GravitySchedule((landing + SWITCH_DELAY_S,), (tilted, g))
        e_clean = restitution_for_edit(scene, states, contacts, index)
        flight = integrate_sphere(p, v, t_switch, radius, schedule, e_clean, states.t[frame:])
        object_id = str(states.object_ids[index])
        fps = float(scene.frame_rate)
        return (
            with_flight(states, index, frame, flight),
            [bounce_record(object_id, b, radius, fps) for b in flight.bounces],
        )


PLUGINS = (GravitySwitch,)
