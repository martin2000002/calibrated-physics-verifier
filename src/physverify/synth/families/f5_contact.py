"""f5: a row of floor bodies touching or nearly touching, and a ball rolling without slipping into
a separate cube."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.schema.scene import ObjectSpec, SceneSpec
from physverify.synth.ballistics import rolling_angular_velocity
from physverify.synth.family_toolkit import (
    FamilyCommon,
    box_envelope,
    checked,
    framed_scene,
    heading,
    image_plane_azimuth,
    material_object,
    next_id,
    sampled_extents,
    sphere_envelope,
    with_retries,
)
from physverify.synth.placement import palette, rest_height

NAME = "f5_contact"
Vec = NDArray[np.float64]


class Params(FamilyCommon):
    settle_frames: int = 6
    cluster_size_range: tuple[int, int] = (2, 4)
    cluster_labels: tuple[str, ...] = ("cube", "cylinder")
    touch_probability: float = 0.5
    gap_range_m: tuple[float, float] = (0.002, 0.03)
    ball_speed_range: tuple[float, float] = (0.3, 0.6)
    # Native frame (after settling) at which the ball reaches the cube.
    hit_frame_range: tuple[float, float] = (8.0, 16.0)
    cluster_clearance_m: float = 0.05
    cluster_extra_range_m: tuple[float, float] = (0.0, 0.1)
    friction: float = 0.6
    restitution: float = 0.2
    plane_spread_deg: float = 15.0


class ContactFamily:
    name: ClassVar[str] = NAME
    Params: ClassVar[type[BaseModel]] = Params

    def sample(self, rng: np.random.Generator, params: BaseModel) -> SceneSpec:
        chosen = checked(params, Params)
        return with_retries(NAME, rng, lambda r: _draw(r, chosen))


def _draw(rng: np.random.Generator, params: Params) -> SceneSpec:
    n_cluster = int(rng.integers(params.cluster_size_range[0], params.cluster_size_range[1] + 1))
    colors = palette(rng, n_cluster + 2)
    motion = heading(float(rng.uniform(0.0, 2.0 * math.pi)))
    azimuth = image_plane_azimuth(motion, rng, params.plane_spread_deg)
    yaw = math.atan2(float(motion[1]), float(motion[0]))
    cube_ext = sampled_extents(params, "cube", rng)
    cube = material_object(
        params, "cube_0", "cube", cube_ext, (0.0, 0.0, rest_height(cube_ext)), colors[0], yaw
    )
    ball, path = _rolling_ball(rng, params, motion, cube_ext[0] / 2.0, colors[1])
    away = _far_side(motion, azimuth)
    lateral_min = max(ball.extents[0], cube_ext[1]) / 2.0 + params.cluster_clearance_m
    cluster = _cluster(rng, params, n_cluster, motion, away, lateral_min, path, colors[2:], yaw)
    objects = (cube, ball, *cluster)
    speed = float(np.linalg.norm(ball.velocity))
    pushed = np.asarray(cube.position) + motion * speed * params.n_frames / params.fps
    envelope = np.vstack(
        [
            box_envelope(cube),
            box_envelope(
                material_object(params, "cube_0", "cube", cube_ext, pushed, colors[0], yaw)
            ),
            sphere_envelope(path, ball.extents[0] / 2.0),
            *[box_envelope(o) for o in cluster],
        ]
    )
    return framed_scene(
        params,
        NAME,
        objects,
        envelope,
        rng,
        settle_frames=params.settle_frames,
        azimuth_deg=azimuth,
    )


def _rolling_ball(
    rng: np.random.Generator,
    params: Params,
    motion: Vec,
    cube_half: float,
    color: tuple[float, float, float],
) -> tuple[ObjectSpec, Vec]:
    """Ball rolling along ``motion`` that reaches the cube at the drawn native frame; the start
    distance includes the settling frames. Returns the ball and its straight path to contact."""
    extents = sampled_extents(params, "ball", rng)
    radius = extents[0] / 2.0
    speed = float(rng.uniform(*params.ball_speed_range))
    hit = float(rng.uniform(*params.hit_frame_range))
    travel = speed * (params.settle_frames + hit) / params.fps
    start = -(radius + cube_half + travel) * motion + radius * np.array([0.0, 0.0, 1.0])
    velocity = speed * motion
    ball = material_object(
        params,
        "ball_0",
        "ball",
        extents,
        start,
        color,
        velocity=velocity,
        angular_velocity=rolling_angular_velocity(velocity, radius),
    )
    path = np.vstack([start, start + travel * motion])
    return ball, path


def _far_side(motion: Vec, azimuth_deg: float) -> Vec:
    """Horizontal unit vector orthogonal to ``motion`` that points away from the camera."""
    away = -heading(math.radians(azimuth_deg))
    side = away - float(away @ motion) * motion
    out: Vec = side / np.linalg.norm(side)
    return out


def _cluster(
    rng: np.random.Generator,
    params: Params,
    n: int,
    motion: Vec,
    side: Vec,
    lateral_min: float,
    path: Vec,
    colors: list[tuple[float, float, float]],
    yaw: float,
) -> list[ObjectSpec]:
    """A row along ``motion`` beside the ball path; neighbours touch (gap 0) or keep a small gap."""
    labels = [
        params.cluster_labels[int(rng.integers(len(params.cluster_labels)))] for _ in range(n)
    ]
    extents = [sampled_extents(params, label, rng) for label in labels]
    gaps = [
        0.0 if rng.uniform() < params.touch_probability else float(rng.uniform(*params.gap_range_m))
        for _ in range(n - 1)
    ]
    along = np.cumsum(
        [0.0] + [(extents[k][0] + extents[k + 1][0]) / 2.0 + gaps[k] for k in range(n - 1)]
    )
    lateral = (
        lateral_min
        + max(e[1] for e in extents) / 2.0
        + float(rng.uniform(*params.cluster_extra_range_m))
    )
    centre = float(rng.uniform(float(path[0] @ motion), 0.0)) - float(along[-1]) / 2.0
    used: list[str] = ["cube_0", "ball_0"]
    bodies: list[ObjectSpec] = []
    for label, ext, offset, color in zip(labels, extents, along, colors, strict=True):
        xy = (centre + float(offset)) * motion + lateral * side
        object_id = next_id(label, used)
        used.append(object_id)
        bodies.append(
            material_object(
                params, object_id, label, ext, (xy[0], xy[1], rest_height(ext)), color, yaw
            )
        )
    return bodies


PLUGINS = (ContactFamily,)
