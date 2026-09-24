"""f1: two balls launched at frame 0, moving nearly in the image plane (ADR 0004 §3.4)."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from pydantic import BaseModel

from physverify.schema.scene import ObjectSpec, SceneSpec
from physverify.synth.layout import palette
from physverify.synth.scene_kit import (
    FamilyCommon,
    RejectedSampleError,
    checked,
    clip_times,
    framed_scene,
    gravity_vector,
    heading,
    image_plane_azimuth,
    make_object,
    min_separation,
    sampled_extents,
    sphere_envelope,
    sphere_path,
    with_retries,
)

NAME = "f1_projectile"
LABEL = "ball"


class Params(FamilyCommon):
    height_range: tuple[float, float] = (0.25, 0.35)
    vz_range: tuple[float, float] = (2.0, 3.0)
    horizontal_speed_range: tuple[float, float] = (0.3, 0.8)
    plane_spread_deg: float = 15.0
    start_gap_range: tuple[float, float] = (0.15, 0.4)
    min_gap_m: float = 0.05
    restitution_range: tuple[float, float] = (0.5, 0.8)
    friction: float = 0.5
    # Gravity scales framed in advance, so r5b_gravity_scale (k = 1 +- 0.4 at L4) stays in view.
    envelope_gravity_scales: tuple[float, ...] = (0.6, 1.0, 1.4)


class ProjectileFamily:
    name: ClassVar[str] = NAME
    Params: ClassVar[type[BaseModel]] = Params

    def sample(self, rng: np.random.Generator, params: BaseModel) -> SceneSpec:
        chosen = checked(params, Params)
        return with_retries(NAME, rng, lambda r: _draw(r, chosen))


def _draw(rng: np.random.Generator, params: Params) -> SceneSpec:
    motion = heading(float(rng.uniform(0.0, 2.0 * math.pi)))
    colors = palette(rng, 2)
    objects: list[ObjectSpec] = []
    start = np.zeros(3)
    for k in range(2):
        extents = sampled_extents(params, LABEL, rng)
        radius = extents[0] / 2.0
        if k == 1:
            start = start + motion * (
                objects[0].extents[0] / 2.0 + radius + rng.uniform(*params.start_gap_range)
            )
        tilt = math.radians(float(rng.uniform(-params.plane_spread_deg, params.plane_spread_deg)))
        direction = _rotated(motion, tilt)
        speed = float(rng.uniform(*params.horizontal_speed_range))
        velocity = speed * direction + np.array([0.0, 0.0, rng.uniform(*params.vz_range)])
        position = np.array([start[0], start[1], radius + rng.uniform(*params.height_range)])
        objects.append(
            make_object(
                f"ball_{k}",
                LABEL,
                extents,
                position,
                colors[k],
                density=params.density,
                friction=params.friction,
                restitution=float(rng.uniform(*params.restitution_range)),
                velocity=velocity,
            )
        )
    envelope = _envelope(objects, params)
    azimuth = image_plane_azimuth(motion, rng, params.plane_spread_deg)
    return framed_scene(params, NAME, objects, envelope, rng, settle_frames=0, azimuth_deg=azimuth)


def _rotated(direction: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * direction[0] - s * direction[1], s * direction[0] + c * direction[1], 0.0])


def _envelope(objects: list[ObjectSpec], params: Params) -> np.ndarray:
    times = clip_times(params)
    points: list[np.ndarray] = []
    for scale in params.envelope_gravity_scales:
        g = scale * gravity_vector()
        paths = [sphere_path(o.position, o.velocity, g, o.extents[0] / 2.0, times) for o in objects]
        gap = min_separation(paths[0], paths[1])
        if gap <= sum(o.extents[0] / 2.0 for o in objects) + params.min_gap_m:
            raise RejectedSampleError(f"balls pass within {gap:.3f} m of each other")
        points.extend(
            sphere_envelope(p, o.extents[0] / 2.0) for p, o in zip(paths, objects, strict=True)
        )
    return np.vstack(points)


PLUGINS = (ProjectileFamily,)
