"""Ray-cast views of one upright solid standing on the ground (world z up), for the LIFT v5
geometry tests: a camera at a given elevation looks at the solid's centre."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from physverify.schema.states import PrimitiveKind

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

FOCAL_PX = 400.0
IMAGE_PX = 160
DISTANCE = 3.0


@dataclass(frozen=True)
class UprightView:
    mask: BoolArray
    z: FloatArray
    intrinsics: FloatArray
    gravity: FloatArray
    rotation: FloatArray
    """Truth local-to-camera rotation."""
    centre: FloatArray
    """Truth centre in camera coordinates."""


def _camera(elevation_deg: float, target: FloatArray) -> tuple[FloatArray, FloatArray]:
    """(camera-to-world rotation with OpenCV axes, camera position) looking at ``target``."""
    e = math.radians(elevation_deg)
    forward = np.array([0.0, math.cos(e), -math.sin(e)])
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(forward, right)
    return np.column_stack([right, down, forward]), target - DISTANCE * forward


def _hit_box(origin: FloatArray, rays: FloatArray, half: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - origin) / rays, (half - origin) / rays
    near = np.nanmax(np.minimum(t1, t2), axis=1)
    far = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((near <= far) & (near > 0), near, np.inf)


def _hit_cylinder(origin: FloatArray, rays: FloatArray, radius: float, half: float) -> FloatArray:
    a = rays[:, 0] ** 2 + rays[:, 1] ** 2
    b = 2.0 * (origin[0] * rays[:, 0] + origin[1] * rays[:, 1])
    c = origin[0] ** 2 + origin[1] ** 2 - radius**2
    disc = b**2 - 4 * a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        side = (-b - np.sqrt(disc)) / (2 * a)
        side_z = origin[2] + side * rays[:, 2]
        side = np.where((disc >= 0) & (np.abs(side_z) <= half) & (side > 0), side, np.inf)
        cap = (np.sign(origin[2]) * half - origin[2]) / rays[:, 2]
    cap_xy = origin[:2] + cap[:, None] * rays[:, :2]
    inside = (np.sum(cap_xy**2, axis=1) <= radius**2) & (cap > 0)
    return np.minimum(side, np.where(inside, cap, np.inf))


def _hit_sphere(origin: FloatArray, rays: FloatArray, radius: float) -> FloatArray:
    b = rays @ origin
    disc = b**2 - (origin @ origin - radius**2)
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def render(
    kind: PrimitiveKind, extents: FloatArray, yaw_deg: float, elevation_deg: float
) -> UprightView:
    """The solid (full ``extents`` along its local axes, local z up) at ``yaw_deg`` about up."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    obj = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    centre = np.array([0.0, 0.0, extents[2] / 2.0])
    world_from_cam, eye = _camera(elevation_deg, centre)
    k = np.array([[FOCAL_PX, 0, IMAGE_PX / 2], [0, FOCAL_PX, IMAGE_PX / 2], [0, 0, 1.0]])
    rows, cols = np.mgrid[0:IMAGE_PX, 0:IMAGE_PX]
    pixels = np.stack([cols + 0.5, rows + 0.5, np.ones_like(cols, dtype=float)], -1).reshape(-1, 3)
    rays_cam = pixels @ np.linalg.inv(k).T
    rays_local = rays_cam @ (obj.T @ world_from_cam).T
    origin = obj.T @ (eye - centre)
    if kind is PrimitiveKind.SPHERE:
        t = _hit_sphere(
            origin, rays_local / np.linalg.norm(rays_local, axis=1)[:, None], extents[0] / 2
        )
        t = t / np.linalg.norm(rays_local, axis=1)
    elif kind is PrimitiveKind.CYLINDER:
        t = _hit_cylinder(origin, rays_local, extents[0] / 2.0, extents[2] / 2.0)
    else:
        t = _hit_box(origin, rays_local, extents / 2.0)
    hit = np.isfinite(t).reshape(IMAGE_PX, IMAGE_PX)
    z = np.where(hit, t.reshape(IMAGE_PX, IMAGE_PX), 0.0)
    cam_from_world = world_from_cam.T
    return UprightView(
        mask=hit,
        z=z,
        intrinsics=k,
        gravity=cam_from_world @ np.array([0.0, 0.0, -1.0]),
        rotation=cam_from_world @ obj,
        centre=cam_from_world @ (centre - eye),
    )
