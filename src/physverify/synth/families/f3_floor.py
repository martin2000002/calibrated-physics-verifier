"""f3_floor: two equal balls rolling on the floor collide at t_c, with no bounce afterwards."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.core.models import expect_params
from physverify.schema.scene import SceneSpec
from physverify.synth.ballistics import energy_scale, rolling_angular_velocity
from physverify.synth.family_toolkit import (
    FamilyCommon,
    RejectedSampleError,
    clip_times,
    framed_scene,
    gravity_vector,
    heading,
    image_plane_azimuth,
    kind_of,
    make_object,
    sampled_extents,
    sphere_envelope,
    sphere_path,
    with_retries,
)
from physverify.synth.placement import palette, primitive_volume

Vec = NDArray[np.float64]
NAME = "f3_floor"
LABEL = "ball"
UP = np.array([0.0, 0.0, 1.0])


class Params(FamilyCommon):
    collision_time_range: tuple[float, float] = (0.30, 0.45)
    mass_ratio_range: tuple[float, float] = (1.0, 3.0)
    relative_speed_range: tuple[float, float] = (1.5, 3.0)
    max_cm_speed: float = 0.3
    # Low so that the floor (restitution 1, combined with the ball's by Bullet) does not bounce
    # a ball back up; the ball-ball restitution is about the product of the two.
    restitution_range: tuple[float, float] = (0.3, 0.5)
    friction: float = 0.1
    min_start_gap_m: float = 0.05
    azimuth_spread_deg: float = 15.0
    # Post-collision speed scale framed in advance: lambda of r3_energy_gain at dE/E = 1.0.
    envelope_energy_gain: float = 1.0


@dataclass(frozen=True)
class FloorPair:
    """Both balls have ``radius``; ``axis`` is horizontal, from ball 1 to ball 0 at contact."""

    radius: float
    masses: tuple[float, float]
    axis: Vec
    v_cm: Vec
    v_rel: float
    t_c: float

    def centres(self) -> tuple[Vec, Vec]:
        """Centres at t_c: the contact point is the origin raised to the rest height."""
        middle = self.radius * UP
        return middle - self.radius * self.axis, middle + self.radius * self.axis


class FloorCollisionFamily:
    name: ClassVar[str] = NAME
    Params: ClassVar[type[BaseModel]] = Params

    def sample(self, rng: np.random.Generator, params: BaseModel) -> SceneSpec:
        chosen = expect_params(params, Params)
        return with_retries(NAME, rng, lambda r: _draw(r, chosen))


def _draw(rng: np.random.Generator, params: Params) -> SceneSpec:
    extents = sampled_extents(params, LABEL, rng)
    ratio = float(rng.uniform(*params.mass_ratio_range))
    mass_b = params.density * primitive_volume(kind_of(LABEL), extents)
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    pair = FloorPair(
        radius=extents[0] / 2.0,
        masses=(ratio * mass_b, mass_b),
        axis=heading(phi),
        v_cm=float(rng.uniform(0.0, params.max_cm_speed))
        * heading(float(rng.uniform(0.0, 2.0 * math.pi))),
        v_rel=float(rng.uniform(*params.relative_speed_range)),
        t_c=float(rng.uniform(*params.collision_time_range)),
    )
    starts = start_positions(pair)
    check_start_gap(pair, starts, params)
    colors = palette(rng, 2)
    restitution = float(rng.uniform(*params.restitution_range))
    densities = (ratio * params.density, params.density)
    objects = [
        make_object(
            f"ball_{k}",
            LABEL,
            extents,
            starts[k],
            colors[k],
            density=densities[k],
            friction=params.friction,
            restitution=restitution,
            velocity=velocity,
            angular_velocity=rolling_angular_velocity(velocity, pair.radius),
        )
        for k, velocity in enumerate(pre_velocities(pair))
    ]
    envelope = _envelope(pair, starts, restitution, params)
    azimuth = image_plane_azimuth(pair.axis, rng, params.azimuth_spread_deg)
    return framed_scene(params, NAME, objects, envelope, rng, settle_frames=0, azimuth_deg=azimuth)


def pre_velocities(pair: FloorPair) -> tuple[Vec, Vec]:
    """v_A = v_cm + (m_B/M) v_rel u, v_B = v_cm - (m_A/M) v_rel u (constant while rolling)."""
    total = sum(pair.masses)
    rel = pair.v_rel * pair.axis
    return pair.v_cm + pair.masses[1] / total * rel, pair.v_cm - pair.masses[0] / total * rel


def post_velocities(pair: FloorPair, e_normal: float) -> tuple[Vec, Vec]:
    """v+ = v_cm - e (v- - v_cm): the normal is the horizontal axis, so v_z stays 0."""
    before = pre_velocities(pair)
    after = [pair.v_cm - e_normal * (v - pair.v_cm) for v in before]
    return after[0], after[1]


def start_positions(pair: FloorPair) -> tuple[Vec, Vec]:
    """Backward along straight rolling lines: p0 = c - v t_c."""
    centres = pair.centres()
    a, b = (c - v * pair.t_c for c, v in zip(centres, pre_velocities(pair), strict=True))
    return a, b


def check_start_gap(pair: FloorPair, starts: tuple[Vec, Vec], params: Params) -> None:
    gap = float(np.linalg.norm(starts[0] - starts[1])) - 2.0 * pair.radius
    if gap < params.min_start_gap_m:
        raise RejectedSampleError("the balls start too close together")


def _envelope(
    pair: FloorPair, starts: tuple[Vec, Vec], ball_restitution: float, params: Params
) -> Vec:
    """Rolling lines before t_c plus the lines after it for the clean restitution (framed with the
    ball's e, above Bullet's e_a e_b) and for the speed scale lambda of the largest framed gain."""
    g = gravity_vector()
    times = clip_times(params)
    before = times[times <= pair.t_c]
    after = times[times > pair.t_c] - pair.t_c
    v_minus = pre_velocities(pair)
    v_plus = post_velocities(pair, ball_restitution)
    centres = pair.centres()
    points = [
        sphere_envelope(sphere_path(starts[k], v_minus[k], g, pair.radius, before), pair.radius)
        for k in range(2)
    ]
    gain = energy_scale(pair.masses, list(v_minus), list(v_plus), params.envelope_energy_gain)
    for scale in (1.0, gain):
        for k in range(2):
            velocity = pair.v_cm + scale * (v_plus[k] - pair.v_cm)
            path = sphere_path(centres[k], velocity, g, pair.radius, after)
            points.append(sphere_envelope(np.vstack([centres[k][None], path]), pair.radius))
    return np.vstack(points)


PLUGINS = (FloorCollisionFamily,)
