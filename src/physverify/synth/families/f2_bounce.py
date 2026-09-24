"""f2: one ball dropped onto the floor and bouncing, plus one cube at rest (ADR 0004 §3.4)."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from pydantic import BaseModel

from physverify.schema.scene import SceneSpec
from physverify.schema.states import PrimitiveKind
from physverify.synth.layout import footprint_radius, palette, rest_height
from physverify.synth.scene_kit import (
    FamilyCommon,
    box_envelope,
    checked,
    clip_times,
    framed_scene,
    heading,
    make_object,
    sampled_extents,
    sphere_envelope,
    with_retries,
)

NAME = "f2_bounce"


class Params(FamilyCommon):
    drop_height_range: tuple[float, float] = (0.25, 0.5)
    max_horizontal_speed: float = 0.2
    restitution_range: tuple[float, float] = (0.7, 0.9)
    ball_friction: float = 0.5
    cube_friction: float = 0.6
    cube_restitution: float = 0.1
    cube_clearance_m: float = 0.05
    # Perpendicular distance from the ball path, beyond contact plus the clearance.
    cube_distance_range: tuple[float, float] = (0.0, 0.2)
    # Rebound apex framed for e' = 1.5 (r3_restitution_gain at L4): (1.5)^2 h.
    max_restitution: float = 1.5


class BounceFamily:
    name: ClassVar[str] = NAME
    Params: ClassVar[type[BaseModel]] = Params

    def sample(self, rng: np.random.Generator, params: BaseModel) -> SceneSpec:
        chosen = checked(params, Params)
        return with_retries(NAME, rng, lambda r: _draw(r, chosen))


def _draw(rng: np.random.Generator, params: Params) -> SceneSpec:
    colors = palette(rng, 2)
    ball_ext = sampled_extents(params, "ball", rng)
    radius = ball_ext[0] / 2.0
    height = float(rng.uniform(*params.drop_height_range))
    motion = heading(float(rng.uniform(0.0, 2.0 * math.pi)))
    speed = float(rng.uniform(0.0, params.max_horizontal_speed))
    ball = make_object(
        "ball_0",
        "ball",
        ball_ext,
        (0.0, 0.0, radius + height),
        colors[0],
        density=params.density,
        friction=params.ball_friction,
        restitution=float(rng.uniform(*params.restitution_range)),
        velocity=speed * motion,
    )
    duration = float(clip_times(params)[-1])
    travel = speed * duration
    side = heading(
        math.atan2(motion[1], motion[0]) + (math.pi / 2 if rng.uniform() < 0.5 else -math.pi / 2)
    )
    cube_ext = sampled_extents(params, "cube", rng)
    cube_radius = footprint_radius(PrimitiveKind.BOX, cube_ext)
    offset = (
        radius
        + cube_radius
        + float(rng.uniform(*params.cube_distance_range))
        + params.cube_clearance_m
    )
    along = float(rng.uniform(0.0, travel)) * motion
    cube_xy = side * offset + along
    cube = make_object(
        "cube_0",
        "cube",
        cube_ext,
        (cube_xy[0], cube_xy[1], rest_height(cube_ext)),
        colors[1],
        density=params.density,
        friction=params.cube_friction,
        restitution=params.cube_restitution,
        yaw=float(rng.uniform(0.0, math.pi / 2.0)),
    )
    apex = radius + params.max_restitution**2 * height
    column = np.array([[0.0, 0.0, z] for z in (radius, radius + height, apex)])
    path = np.vstack([column, column + travel * motion])
    envelope = np.vstack([sphere_envelope(path, radius), box_envelope(cube)])
    return framed_scene(params, NAME, (ball, cube), envelope, rng, settle_frames=0)


PLUGINS = (BounceFamily,)
