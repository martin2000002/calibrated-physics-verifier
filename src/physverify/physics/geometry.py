"""Collision primitives, poses and the shared vector helpers (the single definition).

Conventions: ``extents`` are full extents along the local axes; a cylinder has its axis on local
z and diameter ``extents[0]``; a sphere has diameter ``extents[0]``; ``unknown`` is treated as a
box. ``Separation.normal`` points from ``b`` to ``a`` and a negative distance is penetration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.schema.states import PrimitiveKind

Vec = NDArray[np.float64]

CYLINDER_SIDES = 16
"""Sides of the inscribed prism that stands in for a cylinder in SAT and face clipping."""

CYLINDER_PRISM_REL_ERROR = 1.0 - math.cos(math.pi / CYLINDER_SIDES)
"""Worst-case prism error relative to the radius (about 1.9 %)."""

EPS = 1e-12


def as_vector(values: ArrayLike, size: int = 3) -> Vec:
    """Coerce to a float64 vector of exactly ``size`` components."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"expected shape ({size},), got {array.shape}")
    return array


def _unit(values: ArrayLike) -> Vec:
    vector = as_vector(values)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < EPS:
        raise ValueError("direction must be finite and non-zero")
    return vector / norm


@dataclass(frozen=True, eq=False)
class Primitive:
    kind: PrimitiveKind
    extents: Vec

    def __post_init__(self) -> None:
        extents = as_vector(self.extents)
        if not bool(np.all(np.isfinite(extents))) or bool(np.any(extents <= 0)):
            raise ValueError("extents must be finite and > 0")
        object.__setattr__(self, "extents", extents)

    @property
    def half(self) -> Vec:
        half: Vec = self.extents / 2.0
        return half

    @property
    def radius(self) -> float:
        return float(self.extents[0]) / 2.0

    @property
    def shape(self) -> str:
        """Geometry used for this primitive: ``sphere``, ``cylinder`` or ``box``."""
        if self.kind in (PrimitiveKind.SPHERE, PrimitiveKind.CYLINDER):
            return str(self.kind.value)
        return str(PrimitiveKind.BOX.value)


@dataclass(frozen=True, eq=False)
class Pose:
    """``rotation`` maps local coordinates to world: world = rotation @ local + position."""

    position: Vec
    rotation: Vec

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", as_vector(self.position))
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation must be 3x3")
        object.__setattr__(self, "rotation", rotation)

    def to_world(self, local: ArrayLike) -> Vec:
        world: Vec = np.asarray(local, dtype=np.float64) @ self.rotation.T + self.position
        return world

    def to_local(self, world: ArrayLike) -> Vec:
        local: Vec = (np.asarray(world, dtype=np.float64) - self.position) @ self.rotation
        return local


@dataclass(frozen=True, eq=False)
class Solid:
    primitive: Primitive
    pose: Pose


@dataclass(frozen=True, eq=False)
class Plane:
    """Half-space ``normal . x <= offset`` (solid below); ``normal`` is the outward unit normal."""

    normal: Vec
    offset: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "normal", _unit(self.normal))

    @classmethod
    def ground(cls, height: float = 0.0, up: ArrayLike = (0.0, 0.0, 1.0)) -> Plane:
        normal = _unit(up)
        return cls(normal=normal, offset=height)


Body = Solid | Plane


@dataclass(frozen=True, eq=False)
class Separation:
    distance: float
    normal: Vec
    point_a: Vec
    point_b: Vec


@dataclass(frozen=True, eq=False)
class Manifold:
    """Contact patch vertices with per-point normals (b to a) and signed distances."""

    points: Vec
    normals: Vec
    distances: Vec


def matrix_from_quaternion(quaternion: ArrayLike) -> Vec:
    """Rotation matrix of a (w, x, y, z) quaternion (normalized first)."""
    q = as_vector(quaternion, 4)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < EPS:
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_from_matrix(rotation: ArrayLike) -> Vec:
    """(w, x, y, z) quaternion with w >= 0 of a rotation matrix."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(m))
    candidates = [trace, m[0, 0], m[1, 1], m[2, 2]]
    k = int(np.argmax(candidates))
    if k == 0:
        s = 2.0 * math.sqrt(1.0 + trace)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i, j, n = k - 1, k % 3, (k + 1) % 3
        s = 2.0 * math.sqrt(1.0 + m[i, i] - m[j, j] - m[n, n])
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (m[n, j] - m[j, n]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + n] = (m[n, i] + m[i, n]) / s
    quaternion = np.array(q)
    return quaternion if quaternion[0] >= 0 else -quaternion


def pose_from_quaternion(position: ArrayLike, quaternion: ArrayLike) -> Pose:
    return Pose(position=as_vector(position), rotation=matrix_from_quaternion(quaternion))


def plane_basis(normal: ArrayLike) -> tuple[Vec, Vec]:
    """Deterministic orthonormal (u, v) spanning the plane orthogonal to ``normal``."""
    n = _unit(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def project_to_plane(points: ArrayLike, normal: ArrayLike) -> NDArray[np.float64]:
    """(n, 2) coordinates of 3D ``points`` in the ``plane_basis`` of ``normal``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    u, v = plane_basis(normal)
    return np.column_stack([pts @ u, pts @ v])


def assumed_upright_pose(position: ArrayLike, gravity_dir: ArrayLike, yaw_axis: ArrayLike) -> Pose:
    """Pose standing upright: local z = -gravity, local x = ``yaw_axis`` made horizontal."""
    z = -_unit(gravity_dir)
    yaw = np.asarray(yaw_axis, dtype=np.float64)
    x = yaw - float(yaw @ z) * z
    norm = float(np.linalg.norm(x))
    x = plane_basis(z)[0] if not math.isfinite(norm) or norm < 1e-9 else x / norm
    return Pose(position=as_vector(position), rotation=np.column_stack([x, np.cross(z, x), z]))


def characteristic_lengths(extents: ArrayLike) -> NDArray[np.float64]:
    """L = (e_x e_y e_z)^(1/3) over the last axis of ``extents`` (NaN propagates)."""
    lengths: NDArray[np.float64] = np.cbrt(np.prod(np.asarray(extents, dtype=np.float64), axis=-1))
    return lengths


def characteristic_length(extents: ArrayLike) -> float:
    return float(characteristic_lengths(as_vector(extents)))
