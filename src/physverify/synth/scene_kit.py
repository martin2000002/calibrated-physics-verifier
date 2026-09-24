"""Building blocks of the scene families: the shared family parameters, seeded retries, object
construction from the size prior, analytic sphere paths and the framed ``SceneSpec``."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel

from physverify.core.errors import ConfigError
from physverify.physics.geometry import Primitive, Solid, pose_from_quaternion
from physverify.schema.constants import STANDARD_GRAVITY
from physverify.schema.scene import ObjectSpec, SceneSpec
from physverify.schema.states import PrimitiveKind
from physverify.synth.layout import (
    KUBASIC_ASSET,
    CameraFraming,
    UnreachableTargetError,
    extents_for,
    footprint_radius,
    frame_camera,
    object_spec,
    sample_characteristic_length,
)
from physverify.synth.settings import SceneCommon

Vec = NDArray[np.float64]
SAMPLING_TRIES = 100
DRAFT_SEED = 0


class RejectedSampleError(Exception):
    """One sampling draw violates a family constraint; the family draws again."""


class FamilyCommon(SceneCommon):
    """Fields every family adds to ``SceneCommon``. Cylinders keep aspect 1 because PyBullet's URDF
    loader only scales uniformly (ADR 0004, Implementation)."""

    density: float = 1000.0
    aspect_range: tuple[float, float] = (1.0, 1.0)
    framing: CameraFraming = CameraFraming()


def checked[M: BaseModel](params: BaseModel, cls: type[M]) -> M:
    if not isinstance(params, cls):
        raise TypeError(f"expected {cls.__name__}, got {type(params).__name__}")
    return params


def with_retries(
    family: str, rng: np.random.Generator, draw: Callable[[np.random.Generator], SceneSpec]
) -> SceneSpec:
    reasons: list[str] = []
    for _ in range(SAMPLING_TRIES):
        try:
            return draw(rng)
        except (RejectedSampleError, UnreachableTargetError) as exc:
            reasons.append(str(exc))
    summary = "; ".join(list(dict.fromkeys(reasons))[:5])
    raise ConfigError(f"{family}: no valid scene after {SAMPLING_TRIES} draws ({summary})")


def gravity_vector() -> Vec:
    return np.array([0.0, 0.0, -STANDARD_GRAVITY])


def clip_times(params: SceneCommon) -> Vec:
    out: Vec = np.arange(params.n_frames, dtype=np.float64) / params.fps
    return out


def kind_of(label: str) -> PrimitiveKind:
    return KUBASIC_ASSET[label][1]


def sampled_extents(
    params: FamilyCommon, label: str, rng: np.random.Generator
) -> tuple[float, float, float]:
    length = sample_characteristic_length(rng, params.size_priors[label])
    return extents_for(kind_of(label), length, rng, params.aspect_range)


def yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """wxyz rotation about +z."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def heading(angle: float) -> Vec:
    return np.array([math.cos(angle), math.sin(angle), 0.0])


def make_object(
    object_id: str,
    label: str,
    extents: Sequence[float],
    position: ArrayLike,
    color: Sequence[float],
    *,
    density: float,
    friction: float,
    restitution: float,
    yaw: float = 0.0,
    velocity: ArrayLike = (0.0, 0.0, 0.0),
    angular_velocity: ArrayLike = (0.0, 0.0, 0.0),
) -> ObjectSpec:
    return object_spec(
        object_id,
        label,
        extents,
        density,
        np.asarray(position, dtype=np.float64).tolist(),
        friction=friction,
        restitution=restitution,
        color=color,
        velocity=np.asarray(velocity, dtype=np.float64).tolist(),
        angular_velocity=np.asarray(angular_velocity, dtype=np.float64).tolist(),
        quaternion=yaw_quaternion(yaw),
    )


def solid_of(obj: ObjectSpec, position: ArrayLike | None = None) -> Solid:
    """The object's primitive at its spec pose (or at ``position`` with the same orientation)."""
    centre = obj.position if position is None else position
    return Solid(
        Primitive(obj.primitive, np.asarray(obj.extents)),
        pose_from_quaternion(centre, obj.quaternion),
    )


def next_id(label: str, used: Sequence[str]) -> str:
    """``label_k`` with the smallest k not in ``used``."""
    return next(f"{label}_{k}" for k in range(len(used) + 1) if f"{label}_{k}" not in used)


def box_envelope(obj: ObjectSpec) -> Vec:
    """Corners of a vertical prism that bounds the object for any yaw."""
    centre = np.asarray(obj.position, dtype=np.float64)
    radius = footprint_radius(obj.primitive, obj.extents)
    half_h = obj.extents[2] / 2.0
    offsets = np.array(
        [
            [sx * radius, sy * radius, sz * half_h]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    out: Vec = centre + offsets
    return out


def sphere_path(p0: ArrayLike, v0: ArrayLike, g: ArrayLike, radius: float, times: ArrayLike) -> Vec:
    """Ballistic centres with z clamped at the resting height: the horizontal motion is kept after
    landing, an upper bound on where a bouncing or rolling ball can be during the clip."""
    t = np.asarray(times, dtype=np.float64)[:, None]
    path: Vec = np.asarray(p0, dtype=np.float64) + np.asarray(v0) * t + 0.5 * np.asarray(g) * t**2
    path[:, 2] = np.maximum(path[:, 2], radius)
    return path


def sphere_envelope(centres: Vec, radius: float) -> Vec:
    offsets = radius * np.vstack([np.eye(3), -np.eye(3)])
    out: Vec = (centres[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    return out


def min_separation(path_a: Vec, path_b: Vec) -> float:
    return float(np.min(np.linalg.norm(path_a - path_b, axis=1)))


def framed_scene(
    params: FamilyCommon,
    family: str,
    objects: Sequence[ObjectSpec],
    envelope: Vec,
    rng: np.random.Generator,
    *,
    settle_frames: int,
    azimuth_deg: float | None = None,
) -> SceneSpec:
    """SceneSpec with a camera that frames ``envelope``; ``scene_id`` and ``seed`` are drafts that
    ``generate`` replaces."""
    camera = frame_camera(rng, envelope, params.resolution, params.framing, azimuth_deg)
    return SceneSpec(
        scene_id=f"{family}-draft",
        family=family,
        seed=DRAFT_SEED,
        n_frames=params.n_frames,
        settle_frames=settle_frames,
        resolution=params.resolution,
        camera=camera,
        objects=tuple(objects),
        frame_rate=params.fps,
        step_rate=params.step_rate,
        gravity=(0.0, 0.0, -STANDARD_GRAVITY),
        spp=params.spp,
        background=params.background,
    )


def image_plane_azimuth(motion: Vec, rng: np.random.Generator, spread_deg: float) -> float:
    """Camera azimuth (degrees) perpendicular to a horizontal ``motion`` within ``spread_deg``."""
    angle = math.degrees(math.atan2(float(motion[1]), float(motion[0])))
    side = 90.0 if rng.uniform() < 0.5 else -90.0
    return angle + side + float(rng.uniform(-spread_deg, spread_deg))
