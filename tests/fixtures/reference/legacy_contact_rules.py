"""The floor-bounce and ball-ball collision rules that the ground-truth migrations replaced."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from physverify.physics.tracks import records_of
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactTable
from physverify.schema.states import ObjectStateTable
from physverify.synth.perturbations._ballistics import impact_velocity
from physverify.synth.perturbations._measure import (
    bounce_crossing,
    bounce_rows,
    collision_row,
    floor_crossing,
)
from physverify.synth.perturbations._state_edit import radius_of


@dataclass(frozen=True)
class BounceCase:
    """A sphere floor bounce: the row k the old ``floor_bounce`` resolved (its t_c = t_k
    fallback when z_k <= r), z_k - r there, and the (k, source row, t_c) it resolves to now."""

    k: int | None
    depth_m: float
    now: tuple[int, int, float] | None

    @property
    def penetrated(self) -> bool:
        return self.k is not None and self.depth_m <= 0.0

    @property
    def status(self) -> str:
        if self.now is None:
            return "none" if self.k is None else "unresolved"
        if self.now[0] != self.k:
            return "moved"
        return "previous_row" if self.now[1] != self.now[0] else "same_row"


def bounce_case(states: ObjectStateTable, index: int, t_guess: float, g: np.ndarray) -> BounceCase:
    radius = radius_of(states, index)
    gz = float(g[2])
    legacy = next(
        (
            k
            for k in bounce_rows(states, index, t_guess)
            if states.position[k, index, 2] <= radius
            or floor_crossing(states, index, k, radius, gz) is not None
        ),
        None,
    )
    depth = math.nan if legacy is None else float(states.position[legacy, index, 2]) - radius
    return BounceCase(legacy, depth, bounce_crossing(states, index, t_guess, g))


def legacy_grounded(object_id: str, contacts: ContactTable, frame: int) -> bool:
    """The old floor rule: any floor record within one frame of the row."""
    return any(
        r.object_b == ENV_ID and math.floor(r.t_start) - 1 <= frame <= math.ceil(r.t_end) + 1
        for r in records_of(object_id, contacts)
    )


def legacy_collision_velocities(
    states: ObjectStateTable, contacts: ContactTable, index: int, t_c: float, g: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    k = collision_row(states, t_c)
    object_id = str(states.object_ids[index])
    out = []
    for row in (k, k + 1):
        v = states.velocity[row, index]
        grounded = legacy_grounded(object_id, contacts, row)
        out.append(v.copy() if grounded else impact_velocity(v, float(states.t[row]), t_c, g))
    return out[0], out[1]


def floor_reach(vz: float, dt: float, g: float) -> float:
    """Upper bound of the vertical distance a body covers between a row and t_c (``dt`` apart)
    in free fall: a flight row whose floor gap is within it may hide an unrecorded bounce."""
    return abs(vz) * abs(dt) + 0.5 * g * dt * dt
