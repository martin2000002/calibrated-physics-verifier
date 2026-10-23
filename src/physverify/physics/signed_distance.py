"""Signed distance between two bodies: exact sphere and plane pairs, SAT on convex polyhedra
(boxes and the prisms that stand in for cylinders). Conventions are those of ``geometry``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from physverify.physics.geometry import (
    CYLINDER_SIDES,
    EPS,
    Body,
    Plane,
    Pose,
    Separation,
    Solid,
    Vec,
)
from physverify.schema.states import PrimitiveKind

_BOX_FACES = ((0, 2, 6, 4), (1, 3, 7, 5), (0, 1, 5, 4), (2, 3, 7, 6), (0, 1, 3, 2), (4, 5, 7, 6))


@dataclass(frozen=True, eq=False)
class Polyhedron:
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


def polyhedron(solid: Solid) -> Polyhedron:
    primitive = solid.primitive
    if primitive.shape == PrimitiveKind.CYLINDER.value:
        local, faces = _prism_template(primitive.radius, float(primitive.half[2]))
    else:
        local, faces = _box_template(primitive.half)
    vertices = solid.pose.to_world(local)
    centroid = vertices.mean(axis=0)
    loops = tuple(np.asarray(face, dtype=np.int64) for face in faces)
    normals = np.array([_face_normal(vertices[loop], centroid) for loop in loops])
    return Polyhedron(vertices, loops, normals, _edge_directions(vertices, loops))


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


def _sat_axes(a: Polyhedron, b: Polyhedron) -> Vec:
    crosses = np.cross(a.edge_dirs[:, None, :], b.edge_dirs[None, :, :]).reshape(-1, 3)
    norms = np.linalg.norm(crosses, axis=1)
    crosses = crosses[norms > 1e-9] / norms[norms > 1e-9, None]
    return np.vstack([b.face_normals, a.face_normals, crosses])


def _polyhedra(a: Solid, b: Solid) -> Separation:
    """SAT distance. Exact penetration depth when overlapping; a lower bound when separated
    and the closest features are not face-aligned (vertex-edge, vertex-vertex)."""
    pa, pb = polyhedron(a), polyhedron(b)
    axes = _sat_axes(pa, pb)
    proj_a, proj_b = pa.vertices @ axes.T, pb.vertices @ axes.T
    a_above = proj_a.min(axis=0) - proj_b.max(axis=0)
    a_below = proj_b.min(axis=0) - proj_a.max(axis=0)
    separation = np.maximum(a_above, a_below)
    best = int(np.argmax(separation > separation.max() - EPS))
    normal = axes[best] if a_above[best] >= a_below[best] else -axes[best]
    return Separation(
        distance=float(separation[best]),
        normal=normal,
        point_a=_support_point(pa.vertices, -normal),
        point_b=_support_point(pb.vertices, normal),
    )


def _sphere_sphere(a: Solid, b: Solid) -> Separation:
    delta = a.pose.position - b.pose.position
    gap = float(np.linalg.norm(delta))
    normal = delta / gap if gap > EPS else np.array([0.0, 0.0, 1.0])
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
    vertices = polyhedron(a).vertices
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
    if radial_norm > EPS:
        point_a = point_a - a.primitive.radius * radial / radial_norm
    height = float(b.normal @ point_a) - b.offset
    return Separation(height, b.normal, point_a, point_a - height * b.normal)


def _sphere_box(a: Solid, b: Solid) -> Separation:
    centre = b.pose.to_local(a.pose.position)
    half = b.primitive.half
    closest = np.clip(centre, -half, half)
    gap = float(np.linalg.norm(centre - closest))
    if gap > EPS:
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
    radial = centre[:2] / rho if rho > EPS else np.array([1.0, 0.0])
    closest = np.array(
        [*(radial * min(rho, radius)), np.clip(centre[2], -half_height, half_height)]
    )
    gap = float(np.linalg.norm(centre - closest))
    if gap > EPS:
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


def shape(body: Body) -> str:
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
    key = (shape(a), shape(b))
    if key in _PAIRS:
        return _PAIRS[key](a, b)
    if key[::-1] in _PAIRS:
        return _flip(_PAIRS[key[::-1]](b, a))
    raise ValueError(f"no signed distance defined for {key}")
