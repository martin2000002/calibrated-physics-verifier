"""f6: one regulation-size ball thrown in the image plane, filmed from the side close enough that
its apparent diameter and its flight are long enough for a known-size scale (R5b's anchor)."""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import numpy as np
from pydantic import BaseModel

from physverify.core.errors import ConfigError
from physverify.core.models import expect_params
from physverify.schema.scene import ObjectSpec, SceneSpec
from physverify.synth.family_toolkit import (
    FamilyCommon,
    RejectedSampleError,
    clip_times,
    framed_scene,
    gravity_vector,
    heading,
    image_plane_azimuth,
    make_object,
    sphere_envelope,
    sphere_path,
    with_retries,
)
from physverify.synth.placement import (
    CameraFraming,
    project,
    spec_intrinsics,
    spec_world_from_camera,
)

NAME = "f6_known_size"
SHAPE_LABEL = "ball"


class Params(FamilyCommon):
    label: str = "basketball"
    # FIBA size 7: circumference 74.9-78.0 cm.
    diameter_range_m: tuple[float, float] = (0.749 / math.pi, 0.780 / math.pi)
    color: tuple[float, float, float] = (0.85, 0.4, 0.1)
    min_diameter_px: float = 24.0
    min_flight_s: float = 0.7
    # The family's own clip, whatever the dataset profile: a 0.7 s flight under k = 0.6 needs two
    # seconds, and 24 px of a regulation ball need 256 px.
    clip_frames: int = 48
    clip_resolution: int = 256
    flight_s_range: tuple[float, float] = (0.7, 0.9)
    height_range: tuple[float, float] = (0.1, 0.3)
    horizontal_speed_range: tuple[float, float] = (0.1, 0.4)
    plane_spread_deg: float = 10.0
    restitution_range: tuple[float, float] = (0.6, 0.8)
    friction: float = 0.5
    framing: CameraFraming = CameraFraming(elevation_deg=(5.0, 15.0))
    # Framed in advance, so r5b_known_size_gravity_scale (k = 1 +- 0.4) stays in view.
    envelope_gravity_scales: tuple[float, ...] = (0.6, 1.0, 1.4)


class KnownSizeFamily:
    name: ClassVar[str] = NAME
    Params: ClassVar[type[BaseModel]] = Params

    def sample(self, rng: np.random.Generator, params: BaseModel) -> SceneSpec:
        chosen = own_clip(expect_params(params, Params))
        if chosen.flight_s_range[0] < chosen.min_flight_s:
            raise ConfigError(f"{NAME}: flight_s_range starts below min_flight_s")
        return with_retries(NAME, rng, lambda r: _draw(r, chosen))


def own_clip(params: Params) -> Params:
    side = params.clip_resolution
    update = {"n_frames": params.clip_frames, "resolution": (side, side)}
    return params.model_copy(update=update)


def _draw(rng: np.random.Generator, params: Params) -> SceneSpec:
    motion = heading(float(rng.uniform(0.0, 2.0 * math.pi)))
    diameter = float(rng.uniform(*params.diameter_range_m))
    radius = diameter / 2.0
    flight = float(rng.uniform(*params.flight_s_range))
    height = float(rng.uniform(*params.height_range))
    speed = float(rng.uniform(*params.horizontal_speed_range))
    velocity = speed * motion + np.array([0.0, 0.0, launch_speed(flight, height)])
    ball = make_object(
        f"{params.label}_0",
        SHAPE_LABEL,
        (diameter, diameter, diameter),
        np.array([0.0, 0.0, radius + height]),
        params.color,
        density=params.density,
        friction=params.friction,
        restitution=float(rng.uniform(*params.restitution_range)),
        velocity=velocity,
    )
    ball = dataclasses.replace(ball, label=params.label)
    paths = envelope_paths(ball, params)
    envelope = np.vstack([sphere_envelope(p, radius) for p in paths])
    azimuth = image_plane_azimuth(motion, rng, params.plane_spread_deg)
    scene = framed_scene(params, NAME, [ball], envelope, rng, settle_frames=0, azimuth_deg=azimuth)
    smallest = min_apparent_diameter(scene, np.vstack(paths), diameter)
    if smallest < params.min_diameter_px:
        raise RejectedSampleError(f"apparent diameter {smallest:.1f} px < {params.min_diameter_px}")
    return scene


def launch_speed(flight_s: float, height: float) -> float:
    """Vertical speed whose ballistic flight from ``height`` above the resting height lasts
    ``flight_s`` under g: height + v T - g T^2 / 2 = 0."""
    g = -float(gravity_vector()[2])
    return (0.5 * g * flight_s**2 - height) / flight_s


def envelope_paths(ball: ObjectSpec, params: Params) -> list[np.ndarray]:
    times = clip_times(params)
    radius = ball.extents[0] / 2.0
    return [
        sphere_path(ball.position, ball.velocity, k * gravity_vector(), radius, times)
        for k in params.envelope_gravity_scales
    ]


def min_apparent_diameter(scene: SceneSpec, centres: np.ndarray, diameter: float) -> float:
    """Smallest f D / z over ``centres``, z the camera depth (the sphere's image is at least this
    wide off axis)."""
    K = spec_intrinsics(scene.camera, scene.resolution)  # noqa: N806
    _, z = project(centres, K, spec_world_from_camera(scene.camera))
    return float(K[0, 0] * diameter / np.max(z))


PLUGINS = (KnownSizeFamily,)
