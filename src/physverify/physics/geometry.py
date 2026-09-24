"""Collision primitives, signed distance and contact manifolds (the single definition).

Conventions: ``extents`` are full extents along the local axes; a cylinder has its axis on local
z and diameter ``extents[0]``; a sphere has diameter ``extents[0]``; ``unknown`` is treated as a
box. ``Separation.normal`` points from ``b`` to ``a`` and a negative distance is penetration.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.schema.constants import MAX_MANIFOLD_POINTS
from physverify.schema.states import PrimitiveKind

Vec = NDArray[np.float64]

CYLINDER_SIDES = 16
"""Sides of the inscribed prism that stands in for a cylinder in SAT and face clipping."""

CYLINDER_PRISM_REL_ERROR = 1.0 - math.cos(math.pi / CYLINDER_SIDES)
"""Worst-case prism error relative to the radius (about 1.9 %)."""

_EPS = 1e-12


def as_vector(values: ArrayLike, size: int = 3) -> Vec:
    """Coerce to a float64 vector of exactly ``size`` components."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"expected shape ({size},), got {array.shape}")
    return array


def _unit(values: ArrayLike) -> Vec:
    vector = as_vector(values)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < _EPS:
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
    if not math.isfinite(norm) or norm < _EPS:
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


# --- convex polyhedra (boxes and cylinder prisms) -------------------------------------------

_BOX_FACES = ((0, 2, 6, 4), (1, 3, 7, 5), (0, 1, 5, 4), (2, 3, 7, 6), (0, 1, 3, 2), (4, 5, 7, 6))


@dataclass(frozen=True, eq=False)
class _Polyhedron:
    vertices: Vec
    faces: tuple[NDArray[np.int64], ...]
    face_normals: Vec
    edge_dirs: Vec


def _box_template(half: Vec) -> tuple[Vec, tuple[tuple[int, ...], ...]]:
    signs = np.array([[(i >> axis & 1) * 2 - 1 for axis in range(3)] for i in range(8)])
    return signs * half, _BOX_FACES


def _prism_template(radius: float, half_height: float) -> tuple[Vec, tuple[tuple[int, ...], ...]]:
    sides = CYLINDER_SIDES
    angles = 2.0 * np.pi * np.arange(sides) / sides
    ring = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    bottom = np.column_stack([ring, np.full(sides, -half_height)])
    top = np.column_stack([ring, np.full(sides, half_height)])
    walls = tuple((i, (i + 1) % sides, sides + (i + 1) % sides, sides + i) for i in range(sides))
    caps = (tuple(range(sides)), tuple(range(sides, 2 * sides)))
    return np.vstack([bottom, top]), walls + caps


def _polyhedron(solid: Solid) -> _Polyhedron:
    primitive = solid.primitive
    if primitive.shape == PrimitiveKind.CYLINDER.value:
        local, faces = _prism_template(primitive.radius, float(primitive.half[2]))
    else:
        local, faces = _box_template(primitive.half)
    vertices = solid.pose.to_world(local)
    centroid = vertices.mean(axis=0)
    loops = tuple(np.asarray(face, dtype=np.int64) for face in faces)
    normals = np.array([_face_normal(vertices[loop], centroid) for loop in loops])
    return _Polyhedron(vertices, loops, normals, _edge_directions(vertices, loops))


def _face_normal(polygon: Vec, centroid: Vec) -> Vec:
    normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
    normal /= np.linalg.norm(normal)
    outward: Vec = normal if float(normal @ (polygon.mean(axis=0) - centroid)) > 0 else -normal
    return outward


def _edge_directions(vertices: Vec, loops: Sequence[NDArray[np.int64]]) -> Vec:
    unique: list[Vec] = []
    for loop in loops:
        for start, end in zip(loop, np.roll(loop, -1), strict=True):
            edge = vertices[end] - vertices[start]
            edge = edge / np.linalg.norm(edge)
            if all(abs(float(edge @ known)) < 1.0 - 1e-9 for known in unique):
                unique.append(edge)
    return np.array(unique)


def _support_point(vertices: Vec, direction: Vec) -> Vec:
    """Mean of the vertices extreme along ``direction`` (a face or edge centre when tied)."""
    heights = vertices @ direction
    extreme = vertices[heights >= heights.max() - 1e-9]
    point: Vec = extreme.mean(axis=0)
    return point


def _sat_axes(a: _Polyhedron, b: _Polyhedron) -> Vec:
    crosses = np.cross(a.edge_dirs[:, None, :], b.edge_dirs[None, :, :]).reshape(-1, 3)
    norms = np.linalg.norm(crosses, axis=1)
    crosses = crosses[norms > 1e-9] / norms[norms > 1e-9, None]
    return np.vstack([b.face_normals, a.face_normals, crosses])


def _polyhedra(a: Solid, b: Solid) -> Separation:
    """SAT distance. Exact penetration depth when overlapping; a lower bound when separated
    and the closest features are not face-aligned (vertex-edge, vertex-vertex)."""
    pa, pb = _polyhedron(a), _polyhedron(b)
    axes = _sat_axes(pa, pb)
    proj_a, proj_b = pa.vertices @ axes.T, pb.vertices @ axes.T
    a_above = proj_a.min(axis=0) - proj_b.max(axis=0)
    a_below = proj_b.min(axis=0) - proj_a.max(axis=0)
    separation = np.maximum(a_above, a_below)
    best = int(np.argmax(separation > separation.max() - _EPS))
    normal = axes[best] if a_above[best] >= a_below[best] else -axes[best]
    return Separation(
        distance=float(separation[best]),
        normal=normal,
        point_a=_support_point(pa.vertices, -normal),
        point_b=_support_point(pb.vertices, normal),
    )


# --- exact pairs ------------------------------------------------------------------------------


def _sphere_sphere(a: Solid, b: Solid) -> Separation:
    delta = a.pose.position - b.pose.position
    gap = float(np.linalg.norm(delta))
    normal = delta / gap if gap > _EPS else np.array([0.0, 0.0, 1.0])
    ra, rb = a.primitive.radius, b.primitive.radius
    return Separation(
        distance=gap - ra - rb,
        normal=normal,
        point_a=a.pose.position - ra * normal,
        point_b=b.pose.position + rb * normal,
    )


def _sphere_plane(a: Solid, b: Plane) -> Separation:
    height = float(b.normal @ a.pose.position) - b.offset
    radius = a.primitive.radius
    return Separation(
        distance=height - radius,
        normal=b.normal,
        point_a=a.pose.position - radius * b.normal,
        point_b=a.pose.position - height * b.normal,
    )


def _box_plane(a: Solid, b: Plane) -> Separation:
    vertices = _polyhedron(a).vertices
    heights = vertices @ b.normal - b.offset
    lowest = int(np.argmin(heights))
    point_a = vertices[lowest]
    return Separation(
        distance=float(heights[lowest]),
        normal=b.normal,
        point_a=point_a,
        point_b=point_a - heights[lowest] * b.normal,
    )


def _cylinder_plane(a: Solid, b: Plane) -> Separation:
    """Closed form: the support point of the true cylinder along -normal."""
    axis = a.pose.rotation[:, 2]
    along = float(b.normal @ axis)
    radial = b.normal - along * axis
    radial_norm = float(np.linalg.norm(radial))
    point_a = a.pose.position - float(a.primitive.half[2]) * np.sign(along) * axis
    if radial_norm > _EPS:
        point_a = point_a - a.primitive.radius * radial / radial_norm
    height = float(b.normal @ point_a) - b.offset
    return Separation(height, b.normal, point_a, point_a - height * b.normal)


def _sphere_box(a: Solid, b: Solid) -> Separation:
    centre = b.pose.to_local(a.pose.position)
    half = b.primitive.half
    closest = np.clip(centre, -half, half)
    gap = float(np.linalg.norm(centre - closest))
    if gap > _EPS:
        local_normal, depth = (centre - closest) / gap, gap
    else:
        margins = half - np.abs(centre)
        axis = int(np.argmin(margins))
        local_normal = np.zeros(3)
        local_normal[axis] = 1.0 if centre[axis] >= 0 else -1.0
        closest = centre.copy()
        closest[axis] = local_normal[axis] * half[axis]
        depth = -float(margins[axis])
    return _sphere_result(a, b.pose, local_normal, depth, closest)


def _sphere_cylinder(a: Solid, b: Solid) -> Separation:
    centre = b.pose.to_local(a.pose.position)
    radius, half_height = b.primitive.radius, float(b.primitive.half[2])
    rho = float(np.hypot(centre[0], centre[1]))
    radial = centre[:2] / rho if rho > _EPS else np.array([1.0, 0.0])
    closest = np.array(
        [*(radial * min(rho, radius)), np.clip(centre[2], -half_height, half_height)]
    )
    gap = float(np.linalg.norm(centre - closest))
    if gap > _EPS:
        return _sphere_result(a, b.pose, (centre - closest) / gap, gap, closest)
    side, cap = radius - rho, half_height - abs(float(centre[2]))
    if side < cap:
        local_normal = np.array([*radial, 0.0])
        closest = np.array([*(radial * radius), centre[2]])
        return _sphere_result(a, b.pose, local_normal, -side, closest)
    sign = 1.0 if centre[2] >= 0 else -1.0
    closest = np.array([centre[0], centre[1], sign * half_height])
    return _sphere_result(a, b.pose, np.array([0.0, 0.0, sign]), -cap, closest)


def _sphere_result(
    a: Solid, pose: Pose, local_normal: Vec, depth: float, closest: Vec
) -> Separation:
    normal = pose.rotation @ local_normal
    radius = a.primitive.radius
    return Separation(
        distance=depth - radius,
        normal=normal,
        point_a=a.pose.position - radius * normal,
        point_b=pose.to_world(closest),
    )


_PAIRS: MappingProxyType[tuple[str, str], Callable[..., Separation]] = MappingProxyType(
    {
        ("sphere", "sphere"): _sphere_sphere,
        ("sphere", "plane"): _sphere_plane,
        ("box", "plane"): _box_plane,
        ("cylinder", "plane"): _cylinder_plane,
        ("sphere", "box"): _sphere_box,
        ("sphere", "cylinder"): _sphere_cylinder,
        ("box", "box"): _polyhedra,
        ("box", "cylinder"): _polyhedra,
        ("cylinder", "box"): _polyhedra,
        ("cylinder", "cylinder"): _polyhedra,
    }
)


def _shape(body: Body) -> str:
    return "plane" if isinstance(body, Plane) else body.primitive.shape


def _flip(separation: Separation) -> Separation:
    return Separation(
        separation.distance, -separation.normal, separation.point_b, separation.point_a
    )


def signed_distance(a: Body, b: Body) -> Separation:
    """Signed distance between two bodies; the normal points from b to a.

    Exact for sphere-sphere, sphere-plane, box-plane, cylinder-plane, sphere-box and
    sphere-cylinder. Box-box and pairs with a cylinder use SAT on a 16-sided prism.
    """
    key = (_shape(a), _shape(b))
    if key in _PAIRS:
        return _PAIRS[key](a, b)
    if key[::-1] in _PAIRS:
        return _flip(_PAIRS[key[::-1]](b, a))
    raise ValueError(f"no signed distance defined for {key}")


# --- contact manifolds -------------------------------------------------------------------------


def contact_manifold(a: Body, b: Body, tol: float) -> Manifold | None:
    """Contact patch of a and b when their distance is <= tol, else None.

    Spheres give one point. A polyhedron on a plane keeps its vertices within ``tol`` of the
    plane. Two polyhedra use reference/incident face clipping (Sutherland-Hodgman). Points lie
    midway between the surfaces and are reduced with ``reduce_manifold``.
    """
    separation = signed_distance(a, b)
    if separation.distance > tol:
        return None
    if "sphere" in (_shape(a), _shape(b)):
        return _single_point(separation)
    if isinstance(b, Plane):
        assert isinstance(a, Solid)
        return _plane_manifold(a, b, tol)
    if isinstance(a, Plane):
        return _flip_manifold(_plane_manifold(b, a, tol))
    return _clipped_manifold(_polyhedron(a), _polyhedron(b), separation, tol)


def _single_point(separation: Separation) -> Manifold:
    point = (separation.point_a + separation.point_b) / 2.0
    return Manifold(point[None, :], separation.normal[None, :], np.array([separation.distance]))


def _flip_manifold(manifold: Manifold) -> Manifold:
    return Manifold(manifold.points, -manifold.normals, manifold.distances)


def _patch(points: Vec, distances: Vec, normal: Vec) -> Manifold:
    """Move points to the midway contact plane and keep at most MAX_MANIFOLD_POINTS."""
    midway = points - (distances / 2.0)[:, None] * normal
    keep = reduce_manifold_indices(midway, normal, MAX_MANIFOLD_POINTS)
    normals = np.tile(normal, (keep.size, 1))
    return Manifold(midway[keep], normals, distances[keep])


def _plane_manifold(a: Solid, b: Plane, tol: float) -> Manifold:
    vertices = _polyhedron(a).vertices
    heights = vertices @ b.normal - b.offset
    near = heights <= max(tol, float(heights.min()) + 1e-9)
    return _patch(vertices[near], heights[near], b.normal)


def _clipped_manifold(a: _Polyhedron, b: _Polyhedron, sep: Separation, tol: float) -> Manifold:
    normal = sep.normal
    face_b = int(np.argmax(b.face_normals @ normal))
    face_a = int(np.argmax(a.face_normals @ -normal))
    b_is_reference = b.face_normals[face_b] @ normal + 1e-9 >= a.face_normals[face_a] @ -normal
    ref, ref_face, incident = (b, face_b, a) if b_is_reference else (a, face_a, b)
    ref_normal = ref.face_normals[ref_face]
    ref_polygon = ref.vertices[ref.faces[ref_face]]
    incident_face = int(np.argmin(incident.face_normals @ ref_normal))
    clipped = _clip_to_face(
        incident.vertices[incident.faces[incident_face]], ref_polygon, ref_normal
    )
    if clipped.shape[0] == 0:
        return _single_point(sep)
    distances = clipped @ ref_normal - float(ref_normal @ ref_polygon[0])
    near = distances <= tol
    if not bool(np.any(near)):
        return _single_point(sep)
    patch = _patch(clipped[near], distances[near], ref_normal)
    return patch if b_is_reference else _flip_manifold(patch)


def _clip_to_face(polygon: Vec, face: Vec, face_normal: Vec) -> Vec:
    centre = face.mean(axis=0)
    for start, end in zip(face, np.roll(face, -1, axis=0), strict=True):
        side = np.cross(end - start, face_normal)
        side /= np.linalg.norm(side)
        if float(side @ (centre - start)) > 0:
            side = -side
        polygon = _clip_polygon(polygon, side, float(side @ start))
        if polygon.shape[0] == 0:
            break
    return polygon


def _clip_polygon(polygon: Vec, normal: Vec, offset: float) -> Vec:
    """Sutherland-Hodgman: keep the part of ``polygon`` with normal . x <= offset."""
    kept: list[Vec] = []
    count = polygon.shape[0]
    for i in range(count):
        previous, current = polygon[i - 1], polygon[i]
        d_prev = float(previous @ normal) - offset
        d_cur = float(current @ normal) - offset
        if (d_prev <= _EPS) != (d_cur <= _EPS) and count > 1:
            kept.append(previous + (current - previous) * (d_prev / (d_prev - d_cur)))
        if d_cur <= _EPS:
            kept.append(current)
    return np.array(kept).reshape(-1, 3)


def convex_hull_indices(points_2d: ArrayLike) -> NDArray[np.int64]:
    """Counter-clockwise hull vertex indices (monotone chain, collinear points dropped)."""
    xy = np.asarray(points_2d, dtype=np.float64)
    span = float(np.ptp(xy, axis=0).max()) if xy.shape[0] else 0.0
    if span < _EPS:
        return np.zeros(min(1, xy.shape[0]), dtype=np.int64)
    tol = _EPS * span * span
    order = sorted(range(xy.shape[0]), key=lambda i: (xy[i, 0], xy[i, 1]))

    def half_hull(indices: list[int]) -> list[int]:
        chain: list[int] = []
        for i in indices:
            while len(chain) >= 2 and cross_2d(xy[chain[-2]], xy[chain[-1]], xy[i]) <= tol:
                chain.pop()
            chain.append(i)
        return chain

    lower, upper = half_hull(order), half_hull(order[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype=np.int64)


def cross_2d(o: Vec, a: Vec, b: Vec) -> float:
    """z-component of (a - o) x (b - o); positive when o, a, b turn counter-clockwise."""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def reduce_manifold_indices(
    points: ArrayLike, normal: ArrayLike, max_points: int = MAX_MANIFOLD_POINTS
) -> NDArray[np.int64]:
    """Indices of the convex hull of ``points`` in the plane orthogonal to ``normal``,
    greedily dropping the vertex that loses the least area until at most ``max_points``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise ValueError("a manifold needs at least one point")
    xy = project_to_plane(pts, normal)
    hull = convex_hull_indices(xy).tolist()
    while len(hull) > max(1, max_points):
        areas = [
            abs(cross_2d(xy[hull[i - 1]], xy[hull[i]], xy[hull[(i + 1) % len(hull)]]))
            for i in range(len(hull))
        ]
        hull.pop(int(np.argmin(areas)))
    return np.array(hull, dtype=np.int64)


def reduce_manifold(
    points: ArrayLike, normal: ArrayLike, max_points: int = MAX_MANIFOLD_POINTS
) -> Vec:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    reduced: Vec = pts[reduce_manifold_indices(pts, normal, max_points)]
    return reduced


def support_polygon(manifolds: Sequence[Manifold], gravity_dir: ArrayLike) -> Vec:
    """Convex hull (in the plane orthogonal to gravity) of every point that supports a body."""
    if not manifolds:
        raise ValueError("a support polygon needs at least one manifold")
    points = np.vstack([m.points for m in manifolds])
    return reduce_manifold(points, gravity_dir, max_points=points.shape[0])


def polygon_area(points: ArrayLike, normal: ArrayLike) -> float:
    """Area of the convex hull of ``points`` projected on the plane orthogonal to ``normal``."""
    projected = project_to_plane(points, normal)
    xy = projected[convex_hull_indices(projected)]
    if xy.shape[0] < 3:
        return 0.0
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))
