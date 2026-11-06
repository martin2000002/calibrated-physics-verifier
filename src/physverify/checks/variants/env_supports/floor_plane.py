"""Allow the floor's RESTING records of two rolling spheres as supports of their impact.

The floor's normal force and gravity act along the floor normal n_f; with every side velocity of
both bodies within ``max_tilt_deg`` of the floor plane (and, on ground truth, the impact normal
too), they leave the pair's momentum along the impact normal unchanged. Floor friction is what is
left: an estimated fraction of a percent of the impulse over the contact, and over the side fits a
one-signed bias that only lowers s (the bodies decelerate along their own velocities, which slows
the separation), so the check stays conservative. That bias is bounded by RelSpeedParams'
``support_bias_budget``, a declared scope and not a run-time gate (it is not measurable without
truth)."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from physverify.checks.variants._image_impulse import FloatArray, free_rows, line_fit
from physverify.checks.variants.env_supports._base import Rows, Support, SupportQuery
from physverify.schema.contacts import ContactKind
from physverify.schema.states import ObjectStateTable, PrimitiveKind


class FloorPlane:
    """Holds the ENV RESTING records of either sphere whose normal is within ``max_tilt_deg`` of
    the impact normal's plane (on ground truth); the event is supported when every named one is
    held, each body has one, no third body touches the pair between its sides and the four side
    velocities, walked without the held records, lie within ``max_tilt_deg`` of every held
    floor."""

    name: ClassVar[str] = "floor_plane"

    def allows(self, states: ObjectStateTable, query: SupportQuery) -> Support | None:
        e = query.event
        if not all(_sphere(states, i) for i in query.pair):
            return None
        normals = [_unit(np.asarray(r.normal, dtype=np.float64)) for r in query.records]
        held = [
            i
            for i, r in enumerate(query.records)
            if r.kind is ContactKind.RESTING
            and (
                e.normal is None or _max_tilt([normals[i]], [_unit(e.normal)]) <= query.max_tilt_deg
            )
        ]
        named = [query.records[i].object_a for i in held if i < query.named]
        if len(named) < query.named or set(named) != {e.a, e.b}:
            return None
        sides = query.sides_without(held)
        directions = _side_directions(states, query, sides)
        if directions is None or _third_body(states, query, sides):
            return None
        floors = [normals[i] for i in held]
        angles = {"velocity_tilt_deg": _max_tilt(floors, directions)}
        if e.normal is not None:
            angles["normal_tilt_deg"] = _max_tilt(floors, [_unit(e.normal)])
        if max(angles.values()) > query.max_tilt_deg:
            return None
        return Support(tuple(held), angles)


def _sphere(states: ObjectStateTable, index: int) -> bool:
    return str(states.primitive[index]) == PrimitiveKind.SPHERE.value


def _third_body(states: ObjectStateTable, query: SupportQuery, sides: Rows) -> bool:
    """A record of ``others`` busy on a row between the last before and the first after row."""
    left, right = sides
    if not (left and right):
        return False
    fi = np.asarray(states.frame_index, dtype=np.int64)
    span = slice(left[-1], right[0] + 1)
    bodies = (query.event.a, query.event.b)
    return any(
        not bool(free_rows(fi, query.others, body, query.margin)[span].all()) for body in bodies
    )


def _side_directions(
    states: ObjectStateTable, query: SupportQuery, sides: Rows
) -> list[FloatArray] | None:
    """Unit 3D velocities of both bodies on both sides (line fits of the positions); None when a
    side has fewer than two rows or a body does not move on it."""
    fi = np.asarray(states.frame_index, dtype=np.float64)
    out = []
    for rows in sides:
        if len(rows) < 2:
            return None
        r = list(rows)
        for index in query.pair:
            p = np.asarray(states.position[r, index], dtype=np.float64)
            v = line_fit(fi[r], p, query.event.t).slope
            size = float(np.linalg.norm(v))
            if not (math.isfinite(size) and size > 0.0):
                return None
            out.append(v / size)
    return out


def _unit(v: FloatArray) -> FloatArray:
    unit: FloatArray = v / np.linalg.norm(v)
    return unit


def _max_tilt(normals: list[FloatArray], directions: list[FloatArray]) -> float:
    """Largest angle (deg) out of the floor plane, asin |n_f . d|, over every pair."""
    dots = [abs(float(n @ d)) for n in normals for d in directions]
    return max(math.degrees(math.asin(min(1.0, x))) for x in dots)


PLUGINS = (FloorPlane,)
