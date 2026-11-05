"""Shared by the R1 edits: the resting pairs a body can be pushed into and the horizontal offset
that makes the pair's penetration reach a target."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from physverify.physics.signed_distance import signed_distance
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactTable
from physverify.schema.states import ObjectStateTable
from physverify.synth.frame_windows import on_top_pairs, rest_mask, solid_at, supports_nothing
from physverify.synth.magnitudes import length_of, penetration_between
from physverify.synth.perturbations._state_edit import Vec, translated
from physverify.synth.placement import UP, solve_offset

SOLVE_SLACK = 1e-6


def penetration_offset(states: ObjectStateTable, a: int, b: int, magnitude: float) -> Vec:
    """Horizontal shift of B towards A after which max_t pen(A, B) / L reaches ``magnitude``."""
    mid = states.n_frames // 2
    solid_a, solid_b = solid_at(states, a, mid), solid_at(states, b, mid)
    towards = states.position[mid, a] - states.position[mid, b]
    towards = towards - float(towards @ UP) * UP
    direction = towards / np.linalg.norm(towards)
    gap = max(0.0, signed_distance(solid_a, solid_b).distance)
    reach = gap + min(length_of(states, a), length_of(states, b))

    def per_frame(shift: float, frames: Sequence[int]) -> list[float]:
        moved = translated(states, b, 0, shift * direction)
        return [penetration_between(solid_at(states, a, t), solid_at(moved, b, t)) for t in frames]

    # The max over frames is solved on a growing set of active frames: a subset
    # under-estimates it, so the frame that exceeds the target joins until none does.
    active = [mid]
    while True:
        shift = solve_offset(lambda s: max(per_frame(s, active)), magnitude, 0.0, reach, tol=1e-6)
        values = per_frame(shift, range(states.n_frames))
        worst = int(np.argmax(values))
        if worst in active or values[worst] <= magnitude + SOLVE_SLACK:
            break
        active.append(worst)
    out: Vec = shift * direction
    return out


def resting_pairs(
    states: ObjectStateTable, contacts: ContactTable, max_gap: float, tol: float
) -> list[tuple[int, int]]:
    """(A, B): both at rest the whole clip on the floor, clean gap <= ``max_gap``, nothing on B."""
    ids = states.object_ids.astype(str).tolist()
    on_floor = {up for up, low in on_top_pairs(contacts) if low == ENV_ID}
    still = [
        i for i, oid in enumerate(ids) if oid in on_floor and bool(np.all(rest_mask(states, i)))
    ]
    mid = states.n_frames // 2
    pairs: list[tuple[int, int]] = []
    for a in still:
        for b in still:
            if a == b or not supports_nothing(ids[b], contacts):
                continue
            gap = signed_distance(solid_at(states, a, mid), solid_at(states, b, mid)).distance
            if -tol <= gap <= max_gap:
                pairs.append((a, b))
    return pairs
