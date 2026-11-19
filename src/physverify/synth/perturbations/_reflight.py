"""Re-flying one sphere analytically from an edited state, with the floor impacts it makes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from physverify.schema.contacts import ContactRecord, ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.ballistics import (
    SWITCH_DELAY_S,
    Flight,
    GravitySchedule,
    integrate_sphere,
    time_to_height,
)
from physverify.synth.perturbations._state_edit import (
    Vec,
    bounce_record,
    radius_of,
    restitution_for_edit,
    with_flight,
)


@dataclass(frozen=True)
class Launch:
    """The state a re-flight starts from, at time ``t`` (s), and the gravity it follows."""

    position: Vec
    velocity: Vec
    t: float
    schedule: GravitySchedule


@dataclass(frozen=True)
class Reflown:
    """``states`` with the new flight written in, its floor impacts and the clean restitution."""

    states: ObjectStateTable
    flight: Flight
    records: list[ContactRecord]
    e_clean: float


def reflight(
    scene: SceneSpec,
    clean: ObjectStateTable,
    contacts: ContactTable,
    index: int,
    start: int,
    launch: Launch,
    target: ObjectStateTable | None = None,
) -> Reflown:
    """Object ``index`` flown from ``launch`` over the frames ``start..`` of ``clean`` (radius and
    restitution read there) and written into ``target`` (``clean`` when None)."""
    radius = radius_of(clean, index)
    e_clean = restitution_for_edit(scene, clean, contacts, index)
    flight = integrate_sphere(
        launch.position,
        launch.velocity,
        launch.t,
        radius,
        launch.schedule,
        e_clean,
        clean.t[start:],
    )
    object_id = str(clean.object_ids[index])
    fps = float(scene.frame_rate)
    records = [bounce_record(object_id, b, radius, fps) for b in flight.bounces]
    moved = with_flight(clean if target is None else target, index, start, flight)
    return Reflown(moved, flight, records, e_clean)


def tilted_until_landing(
    t: float, position: Vec, velocity: Vec, radius: float, gravities: tuple[Vec, Vec]
) -> GravitySchedule | None:
    """The tilted gravity of ``gravities`` from ``t`` until a hair after the first landing (so the
    bounce itself is solved with it), then the clean one; None when the sphere never lands."""
    tilted, g = gravities
    landing = t + time_to_height(float(position[2]), float(velocity[2]), float(tilted[2]), radius)
    if not math.isfinite(landing):
        return None
    return GravitySchedule((landing + SWITCH_DELAY_S,), (tilted, g))
