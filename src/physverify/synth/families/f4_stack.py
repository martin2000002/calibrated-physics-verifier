"""f4: a base cube with one or two bodies stacked on it and a loose body on the floor, all at rest
(ADR 0004 §3.4). The top body is the only candidate of ``r2_unsupported_rest``."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.physics.geometry import characteristic_length, support_polygon
from physverify.physics.measures.support_overhang import overhang_fraction
from physverify.schema.scene import ObjectSpec, SceneSpec
from physverify.synth.family_toolkit import (
    FamilyCommon,
    RejectedSampleError,
    box_envelope,
    checked,
    framed_scene,
    heading,
    kind_of,
    make_object,
    next_id,
    sampled_extents,
    solid_of,
    with_retries,
)
from physverify.synth.perturbations._measure import overhang_on, support_manifolds
from physverify.synth.placement import UP, footprint_radius, palette, rest_height, solve_offset

NAME = "f4_stack"
Vec = NDArray[np.float64]


class Params(FamilyCommon):
    settle_frames: int = 12
    # Clean overhang_fraction of each stacked body on the one below (negative = inside).
    overhang_range: tuple[float, float] = (-0.4, -0.05)
    top_cube_probability: float = 0.5
    third_probability: float = 0.5
    friction: float = 0.8
    restitution: float = 0.1
    loose_labels: tuple[str, ...] = ("cube", "cylinder")
    loose_clearance_m: float = 0.05
    loose_extra_range_m: tuple[float, float] = (0.0, 0.15)
    # Manifold tolerance of the analytic placement (bodies are placed exactly in contact).
    placement_tol_m: float = 1e-4


class StackFamily:
    name: ClassVar[str] = NAME
    Params: ClassVar[type[BaseModel]] = Params

    def sample(self, rng: np.random.Generator, params: BaseModel) -> SceneSpec:
        chosen = checked(params, Params)
        return with_retries(NAME, rng, lambda r: _draw(r, chosen))


def _draw(rng: np.random.Generator, params: Params) -> SceneSpec:
    colors = palette(rng, 4)
    base_ext = max(
        (sampled_extents(params, "cube", rng) for _ in range(2)), key=characteristic_length
    )
    yaw = float(rng.uniform(0.0, math.pi / 2.0))
    base = _body(
        params, "cube_0", "cube", base_ext, (0.0, 0.0, rest_height(base_ext)), colors[0], yaw
    )
    top_label = "cube" if rng.uniform() < params.top_cube_probability else "cylinder"
    top = _stacked(rng, params, next_id(top_label, ["cube_0"]), top_label, base, colors[1])
    stack = [base, top]
    if top_label == "cube" and rng.uniform() < params.third_probability:
        third = _stacked(
            rng, params, next_id("cube", [base.object_id, top.object_id]), "cube", top, colors[2]
        )
        _check_joint_balance(params, base, top, third)
        stack.append(third)
    loose = _loose(rng, params, stack, colors[3])
    objects = (*stack, loose)
    envelope = np.vstack([box_envelope(o) for o in objects])
    return framed_scene(params, NAME, objects, envelope, rng, settle_frames=params.settle_frames)


def _body(
    params: Params,
    object_id: str,
    label: str,
    extents: tuple[float, float, float],
    position: tuple[float, float, float] | Vec,
    color: tuple[float, float, float],
    yaw: float,
) -> ObjectSpec:
    return make_object(
        object_id,
        label,
        extents,
        position,
        color,
        density=params.density,
        friction=params.friction,
        restitution=params.restitution,
        yaw=yaw,
    )


def _yaw_of(obj: ObjectSpec) -> float:
    w, _, _, z = obj.quaternion
    return 2.0 * math.atan2(z, w)


def _stacked(
    rng: np.random.Generator,
    params: Params,
    object_id: str,
    label: str,
    support: ObjectSpec,
    color: tuple[float, float, float],
) -> ObjectSpec:
    """A body resting on ``support`` (same yaw), shifted along one of its face normals until its
    clean overhang equals a draw from ``overhang_range``."""
    extents = sampled_extents(params, label, rng)
    yaw = _yaw_of(support)
    direction = heading(yaw + math.pi / 2.0 * int(rng.integers(4)))
    top_z = support.position[2] + support.extents[2] / 2.0 + extents[2] / 2.0
    centre = np.array([support.position[0], support.position[1], top_z])
    probe = _body(params, object_id, label, extents, centre, color, yaw)
    lower = solid_of(support)

    def overhang(shift: float) -> float:
        return overhang_on(
            solid_of(probe, centre + shift * direction), [lower], params.placement_tol_m
        )

    target = float(rng.uniform(*params.overhang_range))
    shift = solve_offset(overhang, target, 0.0, support.extents[0] / 2.0)
    return _body(params, object_id, label, extents, centre + shift * direction, color, yaw)


def _check_joint_balance(
    params: Params, base: ObjectSpec, top: ObjectSpec, third: ObjectSpec
) -> None:
    """The joint centre of mass of the two upper bodies must stay inside the base support."""
    masses = np.array([top.mass, third.mass])
    com = masses @ np.array([top.position, third.position]) / masses.sum()
    manifolds = support_manifolds(solid_of(top), [solid_of(base)], params.placement_tol_m)
    if not manifolds:
        raise RejectedSampleError("the second body lost its support manifold")
    polygon = support_polygon(manifolds, -UP)
    if overhang_fraction(com, polygon, -UP, top.extents) > params.overhang_range[1]:
        raise RejectedSampleError("the joint centre of mass of the upper bodies overhangs the base")


def _loose(
    rng: np.random.Generator,
    params: Params,
    stack: list[ObjectSpec],
    color: tuple[float, float, float],
) -> ObjectSpec:
    """A body on the floor at least ``loose_clearance_m`` beyond every footprint of the stack."""
    label = params.loose_labels[int(rng.integers(len(params.loose_labels)))]
    extents = sampled_extents(params, label, rng)
    reach = max(
        float(np.hypot(o.position[0], o.position[1])) + footprint_radius(o.primitive, o.extents)
        for o in stack
    )
    distance = (
        reach
        + footprint_radius(kind_of(label), extents)
        + params.loose_clearance_m
        + float(rng.uniform(*params.loose_extra_range_m))
    )
    xy = distance * heading(float(rng.uniform(0.0, 2.0 * math.pi)))
    object_id = next_id(label, [o.object_id for o in stack])
    position = (float(xy[0]), float(xy[1]), rest_height(extents))
    return _body(
        params, object_id, label, extents, position, color, float(rng.uniform(0.0, math.pi / 2.0))
    )


PLUGINS = (StackFamily,)
