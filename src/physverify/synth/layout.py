"""Scene layout helpers shared by the families: sizes from the prior, masses, placement on the
floor, the camera that frames an envelope of points, and a bracketed 1D solver."""

from __future__ import annotations

import colorsys
import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field

from physverify.config.models import FrozenModel, SizePrior
from physverify.schema.scene import AssetSource, CameraSpec, ObjectSpec
from physverify.schema.states import PrimitiveKind

Vec = NDArray[np.float64]

KUBASIC_ASSET: dict[str, tuple[str, PrimitiveKind]] = {
    "cube": ("cube", PrimitiveKind.BOX),
    "ball": ("sphere", PrimitiveKind.SPHERE),
    "cylinder": ("cylinder", PrimitiveKind.CYLINDER),
}
UP: Vec = np.array([0.0, 0.0, 1.0])


class UnreachableTargetError(ValueError):
    """``solve_offset`` could not bracket the requested value."""


class CameraFraming(FrozenModel):
    elevation_deg: tuple[float, float] = (15.0, 35.0)
    focal_length_mm: float = 35.0
    sensor_width_mm: float = 32.0
    fill: float = Field(default=0.8, gt=0.0, le=1.0)
    max_distance_m: float = 50.0


def lognormal_params(prior: SizePrior) -> tuple[float, float]:
    """(mu, sigma) of ln L so that E[L] = mean_m and std(L) / E[L] = rel_spread."""
    sigma2 = math.log1p(prior.rel_spread**2)
    return math.log(prior.mean_m) - sigma2 / 2.0, math.sqrt(sigma2)


def sample_characteristic_length(rng: np.random.Generator, prior: SizePrior) -> float:
    mu, sigma = lognormal_params(prior)
    return float(rng.lognormal(mu, sigma)) if sigma > 0 else prior.mean_m


def extents_for(
    kind: PrimitiveKind, length: float, rng: np.random.Generator, aspect_range: tuple[float, float]
) -> tuple[float, float, float]:
    """Full extents whose characteristic length (e_x e_y e_z)^(1/3) is exactly ``length``.
    Cylinders stand on their axis with aspect h/d drawn from ``aspect_range``."""
    if kind is PrimitiveKind.CYLINDER:
        aspect = (
            float(rng.uniform(*aspect_range))
            if aspect_range[1] > aspect_range[0]
            else aspect_range[0]
        )
        diameter = length / math.cbrt(aspect)
        return (diameter, diameter, aspect * diameter)
    return (length, length, length)


def primitive_volume(kind: PrimitiveKind, extents: Sequence[float]) -> float:
    ex, ey, ez = (float(e) for e in extents)
    if kind is PrimitiveKind.SPHERE:
        return math.pi * ex**3 / 6.0
    if kind is PrimitiveKind.CYLINDER:
        return math.pi * ex * ey * ez / 4.0
    return ex * ey * ez


def mass_for(kind: PrimitiveKind, extents: Sequence[float], density: float) -> float:
    return density * primitive_volume(kind, extents)


def rest_height(extents: Sequence[float]) -> float:
    """Height of the centre of an upright primitive resting on the floor."""
    return float(extents[2]) / 2.0


def footprint_radius(kind: PrimitiveKind, extents: Sequence[float]) -> float:
    """Radius of a vertical cylinder that bounds the primitive (any yaw)."""
    if kind is PrimitiveKind.BOX:
        return math.hypot(float(extents[0]), float(extents[1])) / 2.0
    return float(extents[0]) / 2.0


def object_spec(
    object_id: str,
    label: str,
    extents: Sequence[float],
    density: float,
    position: Sequence[float],
    *,
    friction: float,
    restitution: float,
    color: Sequence[float],
    velocity: Sequence[float] = (0.0, 0.0, 0.0),
    angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    quaternion: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
) -> ObjectSpec:
    asset_id, kind = KUBASIC_ASSET[label]
    ext = tuple(float(e) for e in extents)
    return ObjectSpec(
        object_id=object_id,
        label=label,
        asset_source=AssetSource.KUBASIC,
        asset_id=asset_id,
        primitive=kind,
        extents=(ext[0], ext[1], ext[2]),
        mass=mass_for(kind, ext, density),
        friction=float(friction),
        restitution=float(restitution),
        static=False,
        position=_triple(position),
        quaternion=_quad(quaternion),
        velocity=_triple(velocity),
        angular_velocity=_triple(angular_velocity),
        color=_triple(color),
    )


def palette(rng: np.random.Generator, n: int) -> list[tuple[float, float, float]]:
    """``n`` well separated saturated colours (evenly spaced hues, random offset)."""
    offset = float(rng.uniform())
    return [colorsys.hsv_to_rgb((offset + i / n) % 1.0, 0.75, 0.85) for i in range(n)]


def camera_axes(position: ArrayLike, look_at: ArrayLike) -> Vec:
    """world_from_camera rotation (columns = OpenCV x right, y down, z forward) for a camera
    whose up vector is world +z."""
    forward = np.asarray(look_at, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, UP)
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(forward, right), forward])


def spec_intrinsics(camera: CameraSpec, resolution: tuple[int, int]) -> Vec:
    """OpenCV K of a CameraSpec with square pixels and the principal point at the centre."""
    width, height = resolution
    f = camera.focal_length_mm / camera.sensor_width_mm * width
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def spec_world_from_camera(camera: CameraSpec) -> Vec:
    pose = np.eye(4)
    pose[:3, :3] = camera_axes(camera.position, camera.look_at)
    pose[:3, 3] = camera.position
    return pose


def project(points: ArrayLike, K: Vec, world_from_camera: Vec) -> tuple[Vec, Vec]:  # noqa: N803
    """Pixel coordinates (n, 2) and camera depth z (n,) of world points (OpenCV convention)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rotation, origin = world_from_camera[:3, :3], world_from_camera[:3, 3]
    cam = (pts - origin) @ rotation
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (cam[:, :2] / z[:, None]) * np.diag(K)[:2] + K[:2, 2]
    return uv, z


def frame_camera(
    rng: np.random.Generator,
    envelope: ArrayLike,
    image_size: tuple[int, int],
    framing: CameraFraming,
    azimuth_deg: float | None = None,
) -> CameraSpec:
    """Camera looking at the centre of ``envelope`` from a random elevation, at the smallest
    distance that keeps every envelope point inside the central ``fill`` of the image."""
    points = np.asarray(envelope, dtype=np.float64).reshape(-1, 3)
    target = (points.min(axis=0) + points.max(axis=0)) / 2.0
    elevation = math.radians(float(rng.uniform(*framing.elevation_deg)))
    azimuth = math.radians(float(rng.uniform(0.0, 360.0)) if azimuth_deg is None else azimuth_deg)
    direction = np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ]
    )

    def spec_at(distance: float) -> CameraSpec:
        return CameraSpec(
            position=_triple(target + distance * direction),
            look_at=_triple(target),
            focal_length_mm=framing.focal_length_mm,
            sensor_width_mm=framing.sensor_width_mm,
        )

    def overflow(distance: float) -> float:
        return _overflow(points, spec_at(distance), image_size, framing.fill)

    radius = float(np.max(np.linalg.norm(points - target, axis=1)))
    lo, hi = max(radius * 1.05, 0.05), framing.max_distance_m
    if overflow(hi) > 0:
        raise UnreachableTargetError("envelope does not fit even at the maximum camera distance")
    if overflow(lo) <= 0:
        return spec_at(lo)
    distance = solve_offset(overflow, 0.0, lo, hi, tol=1e-5)
    return spec_at(min(hi, distance * 1.001))


def _overflow(points: Vec, camera: CameraSpec, size: tuple[int, int], fill: float) -> float:
    """Largest normalized distance beyond the allowed central box (<= 0 means everything fits)."""
    K = spec_intrinsics(camera, size)  # noqa: N806
    uv, z = project(points, K, spec_world_from_camera(camera))
    if bool(np.any(z <= 1e-6)):
        return math.inf
    half = np.array(size, dtype=np.float64) / 2.0
    return float(np.max(np.abs(uv - half) / (fill * half))) - 1.0


def solve_offset(
    f: Callable[[float], float],
    target: float,
    lo: float,
    hi: float,
    *,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> float:
    """s in [lo, hi] with f(s) = target by bisection. ``f`` must cross ``target`` once in the
    interval; if the endpoints do not bracket it, ``UnreachableTargetError``."""
    f_lo, f_hi = f(lo) - target, f(hi) - target
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)) or f_lo * f_hi > 0:
        raise UnreachableTargetError(f"target {target} not bracketed on [{lo}, {hi}]")
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = f(mid) - target
        if f_mid == 0 or (hi - lo) / 2.0 < tol:
            return mid
        if (f_mid < 0) == (f_lo < 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _triple(values: ArrayLike) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=np.float64).reshape(3)
    return (float(v[0]), float(v[1]), float(v[2]))


def _quad(values: ArrayLike) -> tuple[float, float, float, float]:
    v = np.asarray(values, dtype=np.float64).reshape(4)
    return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
