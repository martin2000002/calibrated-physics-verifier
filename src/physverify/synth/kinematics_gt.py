"""The ground-truth velocity convention of simulated clips (ADR 0004 amendment, WP-G item 1).

PyBullet integrates semi-implicit Euler with step h: v_{n+1} = v_n + h g, x_{n+1} = x_n + h v_{n+1}.
In free flight the positions on the step grid then lie exactly on the parabola
x(t_n + m h) = x_n + (v_n + g h / 2) m h + g (m h)^2 / 2, so the velocity the rendered positions
imply is the state velocity plus g h / 2. The "position" convention stores that velocity; resting
contact (the constraint cancels gravity, positions are still) keeps the simulator's.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from numpy.typing import NDArray

from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactTable
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable

Mask = NDArray[np.bool_]


def euler_velocity_offset(scene: SceneSpec) -> NDArray[np.float64]:
    """g h / 2 with the scene's own gravity (k g under R5b) and step h = 1 / step_rate."""
    offset: NDArray[np.float64] = np.asarray(scene.gravity, dtype=np.float64) * (
        0.5 / scene.step_rate
    )
    return offset


def resting_cells(contacts: ContactTable, object_ids: list[str], n_frames: int) -> Mask:
    """(T, K) cells inside a RESTING record of the object, frames floor(t_start)..ceil(t_end)."""
    mask = np.zeros((n_frames, len(object_ids)), dtype=bool)
    for record in contacts.records:
        if record.kind is not ContactKind.RESTING:
            continue
        lo = max(0, math.floor(record.t_start))
        hi = min(n_frames - 1, math.ceil(record.t_end))
        for object_id in (record.object_a, record.object_b):
            if object_id != ENV_ID and object_id in object_ids:
                mask[lo : hi + 1, object_ids.index(object_id)] = True
    return mask


def corrected_cells(states: ObjectStateTable, contacts: ContactTable, scene: SceneSpec) -> Mask:
    """Cells whose velocity ``position_consistent`` shifts: finite, movable and not resting."""
    ids = states.object_ids.astype(str).tolist()
    static = {o.object_id for o in scene.objects if o.static}
    movable = np.array([object_id not in static for object_id in ids], dtype=bool)
    finite = np.isfinite(states.velocity).all(axis=2)
    out: Mask = finite & movable[None, :] & ~resting_cells(contacts, ids, states.n_frames)
    return out


def position_consistent(
    states: ObjectStateTable, contacts: ContactTable, scene: SceneSpec
) -> ObjectStateTable:
    """``states`` (simulator convention) with every non-resting velocity shifted by g h / 2.

    ``contacts`` must be the event table computed on the simulator's own velocities; only the
    velocity array changes, so poses, sizes and visibility (the render inputs) are untouched."""
    velocity = states.velocity.copy()
    velocity[corrected_cells(states, contacts, scene)] += euler_velocity_offset(scene)
    return dataclasses.replace(states, velocity=velocity)
