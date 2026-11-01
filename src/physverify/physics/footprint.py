"""Outline points of a primitive: what its projection along any direction is the hull of."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.physics.geometry import CYLINDER_SIDES, as_vector, matrix_from_quaternion
from physverify.schema.states import PrimitiveKind

FloatArray = NDArray[np.float64]

_CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)


def outline_points(
    center: ArrayLike,
    extents: ArrayLike,
    quaternion: ArrayLike,
    shape: str,
    n_rim: int = CYLINDER_SIDES,
) -> FloatArray:
    """(n, 3) world points whose convex hull is the solid (a box: its 8 corners; a cylinder:
    ``n_rim`` points on each cap rim, axis on local z, diameter ``extents[0]``). A sphere gives
    its centre alone: it rests on a point, so its outline is no support polygon. ``unknown`` is
    a box, as in ``geometry``."""
    position, size = as_vector(center), as_vector(extents)
    if not (bool(np.all(np.isfinite(position))) and bool(np.all(np.isfinite(size)))):
        raise ValueError("center and extents must be finite")
    if shape == PrimitiveKind.SPHERE.value:
        return position[None, :]
    half = size / 2.0
    local = _rims(half, n_rim) if shape == PrimitiveKind.CYLINDER.value else _CORNERS * half
    world: FloatArray = position + local @ matrix_from_quaternion(quaternion).T
    return world


def _rims(half: FloatArray, n_rim: int) -> FloatArray:
    if n_rim < 3:
        raise ValueError("a rim needs at least 3 points")
    angle = np.linspace(0.0, 2.0 * math.pi, n_rim, endpoint=False)
    ring = half[0] * np.column_stack([np.cos(angle), np.sin(angle)])
    caps = [np.column_stack([ring, np.full(n_rim, z)]) for z in (-half[2], half[2])]
    return np.concatenate(caps)


def support_order(
    object_a: str, object_b: str, normal: ArrayLike, up: ArrayLike, max_tilt_deg: float
) -> tuple[str, str] | None:
    """(upper, lower) of a contact whose normal (from b to a) lies within ``max_tilt_deg`` of
    the vertical ``up``, either way; None for a steeper or degenerate normal."""
    n, u = np.asarray(normal, dtype=np.float64), np.asarray(up, dtype=np.float64)
    lean = float(n @ u) / float(np.linalg.norm(n) * np.linalg.norm(u))
    cos_tilt = math.cos(math.radians(max_tilt_deg))
    if lean >= cos_tilt:
        return object_a, object_b
    if lean <= -cos_tilt:
        return object_b, object_a
    return None
