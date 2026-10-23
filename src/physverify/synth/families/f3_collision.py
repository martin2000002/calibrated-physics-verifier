"""f3: two balls launched so that they collide in the air at t_c (ADR 0004 §3.4)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from physverify.schema.scene import SceneSpec
from physverify.synth.family_toolkit import (
    FamilyCommon,
    RejectedSampleError,
    checked,
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
from physverify.synth.perturbations._ballistics import energy_scale
from physverify.synth.placement import palette, primitive_volume

Vec = NDArray[np.float64]
NAME = "f3_collision"
LABEL = "ball"


class Params(FamilyCommon):
    collision_time_range: tuple[float, float] = (0.30, 0.45)
    collision_height_range: tuple[float, float] = (0.35, 0.6)
    mass_ratio_range: tuple[float, float] = (1.0, 3.0)
    relative_speed_range: tuple[float, float] = (1.5, 3.0)
    axis_tilt_deg: float = 15.0
    max_cm_speed: float = 0.3
    # Height of the centre of mass at frame 0; it fixes the vertical centre-of-mass velocity at t_c
    # so both balls are thrown up from above the floor and meet while descending.
    start_height_range: tuple[float, float] = (0.25, 0.45)
    restitution_range: tuple[float, float] = (0.8, 0.95)
    friction: float = 0.1
    min_start_clearance_m: float = 0.05
    # Post-collision speed scale framed in advance: lambda of r3_energy_gain at dE/E = 1.0.
    envelope_energy_gain: float = 1.0


@dataclass(frozen=True)
class Pair:
    radii: tuple[float, float]
    masses: tuple[float, float]
    axis: Vec
    contact: Vec
    v_cm: Vec
    v_rel: float
    t_c: float


class CollisionFamily:
    name: ClassVar[str] = NAME
    Params: ClassVar[type[BaseModel]] = Params

    def sample(self, rng: np.random.Generator, params: BaseModel) -> SceneSpec:
        chosen = checked(params, Params)
        return with_retries(NAME, rng, lambda r: _draw(r, chosen))


def _draw(rng: np.random.Generator, params: Params) -> SceneSpec:
    extents = [sampled_extents(params, LABEL, rng) for _ in range(2)]
    ratio = float(rng.uniform(*params.mass_ratio_range))
    mass_b = params.density * primitive_volume(kind_of(LABEL), extents[1])
    masses = (ratio * mass_b, mass_b)
    densities = (masses[0] / primitive_volume(kind_of(LABEL), extents[0]), params.density)
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    tilt = math.radians(float(rng.uniform(-params.axis_tilt_deg, params.axis_tilt_deg)))
    axis = math.cos(tilt) * heading(phi) + np.array([0.0, 0.0, math.sin(tilt)])
    cm_dir = heading(float(rng.uniform(0.0, 2.0 * math.pi)))
    contact = np.array([0.0, 0.0, rng.uniform(*params.collision_height_range)])
    t_c = float(rng.uniform(*params.collision_time_range))
    start_height = float(rng.uniform(*params.start_height_range))
    pair = Pair(
        radii=(extents[0][0] / 2.0, extents[1][0] / 2.0),
        masses=masses,
        axis=axis,
        contact=contact,
        v_cm=float(rng.uniform(0.0, params.max_cm_speed)) * cm_dir
        + _vertical_cm(contact[2], start_height, t_c),
        v_rel=float(rng.uniform(*params.relative_speed_range)),
        t_c=t_c,
    )
    starts, launches = _launch(pair, params)
    colors = palette(rng, 2)
    restitution = float(rng.uniform(*params.restitution_range))
    objects = [
        make_object(
            f"ball_{k}",
            LABEL,
            extents[k],
            starts[k],
            colors[k],
            density=densities[k],
            friction=params.friction,
            restitution=restitution,
            velocity=launches[k],
        )
        for k in range(2)
    ]
    envelope = _envelope(pair, starts, launches, restitution**2, params)
    azimuth = image_plane_azimuth(heading(phi), rng, params.axis_tilt_deg)
    return framed_scene(params, NAME, objects, envelope, rng, settle_frames=0, azimuth_deg=azimuth)


def _vertical_cm(contact_z: float, start_z: float, t_c: float) -> Vec:
    """Vertical velocity at t_c of a ballistic point that goes from ``start_z`` to ``contact_z``."""
    g_z = float(gravity_vector()[2])
    return np.array([0.0, 0.0, (contact_z - start_z + 0.5 * g_z * t_c**2) / t_c])


def collision_velocities(pair: Pair) -> tuple[Vec, Vec]:
    """v_A = v_cm + (m_B/M) v_rel u, v_B = v_cm - (m_A/M) v_rel u at t_c."""
    total = sum(pair.masses)
    rel = pair.v_rel * pair.axis
    return pair.v_cm + pair.masses[1] / total * rel, pair.v_cm - pair.masses[0] / total * rel


def _launch(pair: Pair, params: Params) -> tuple[list[Vec], list[Vec]]:
    """Backward from the collision: v0 = v(t_c) - g t_c, p0 = c - v0 t_c - g t_c^2 / 2."""
    g = gravity_vector()
    centres = (pair.contact - pair.radii[0] * pair.axis, pair.contact + pair.radii[1] * pair.axis)
    starts: list[Vec] = []
    launches: list[Vec] = []
    for centre, velocity, radius in zip(
        centres, collision_velocities(pair), pair.radii, strict=True
    ):
        v0 = velocity - g * pair.t_c
        p0 = centre - v0 * pair.t_c - 0.5 * g * pair.t_c**2
        if p0[2] < radius + params.min_start_clearance_m:
            raise RejectedSampleError("a ball would start too low")
        starts.append(p0)
        launches.append(v0)
    _check_approach(pair, starts, launches, params)
    return starts, launches


def _check_approach(pair: Pair, starts: list[Vec], launches: list[Vec], params: Params) -> None:
    g = gravity_vector()
    times = np.arange(0.0, pair.t_c, 1.0 / (8.0 * params.fps))
    paths = [
        p + np.outer(times, v) + 0.5 * np.outer(times**2, g)
        for p, v in zip(starts, launches, strict=True)
    ]
    for path, radius in zip(paths, pair.radii, strict=True):
        if np.any(path[:, 2] < radius):
            raise RejectedSampleError("a ball touches the floor before the collision")
    early = times < pair.t_c - 1.0 / (4.0 * params.fps)
    gaps = np.linalg.norm(paths[0][early] - paths[1][early], axis=1)
    if np.any(gaps <= sum(pair.radii)):
        raise RejectedSampleError("the balls touch before t_c")


def _envelope(
    pair: Pair, starts: list[Vec], launches: list[Vec], e_pair: float, params: Params
) -> Vec:
    """Pre-collision arcs plus post-collision arcs for the clean restitution and for the speed
    scale lambda of the largest framed energy gain."""
    g = gravity_vector()
    times = clip_times(params)
    before = times[times <= pair.t_c]
    after = times[times > pair.t_c] - pair.t_c
    v_minus = collision_velocities(pair)
    v_plus = [pair.v_cm + math.sqrt(e_pair) * (pair.v_cm - v) for v in v_minus]
    centres = (pair.contact - pair.radii[0] * pair.axis, pair.contact + pair.radii[1] * pair.axis)
    points = []
    for k in range(2):
        points.append(
            sphere_envelope(
                sphere_path(starts[k], launches[k], g, pair.radii[k], before), pair.radii[k]
            )
        )
    for scale in (1.0, energy_scale(pair.masses, v_minus, v_plus, params.envelope_energy_gain)):
        for k in range(2):
            velocity = pair.v_cm + scale * (v_plus[k] - pair.v_cm)
            path = sphere_path(centres[k], velocity, g, pair.radii[k], after)
            points.append(sphere_envelope(np.vstack([centres[k][None], path]), pair.radii[k]))
    return np.vstack(points)


PLUGINS = (CollisionFamily,)
