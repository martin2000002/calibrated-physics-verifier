"""Contact manifolds of two bodies (face clipping), their reduction to at most
``MAX_MANIFOLD_POINTS`` and the support polygon and area built from them."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.physics.geometry import (
    EPS,
    Body,
    Manifold,
    Plane,
    Separation,
    Solid,
    Vec,
    project_to_plane,
)
from physverify.physics.signed_distance import Polyhedron, polyhedron, shape, signed_distance
from physverify.schema.constants import MAX_MANIFOLD_POINTS


def contact_manifold(a: Body, b: Body, tol: float) -> Manifold | None:
    """Contact patch of a and b when their distance is <= tol, else None.

    Spheres give one point. A polyhedron on a plane keeps its vertices within ``tol`` of the
    plane. Two polyhedra use reference/incident face clipping (Sutherland-Hodgman). Points lie
    midway between the surfaces and are reduced with ``reduce_manifold``.
    """
    separation = signed_distance(a, b)
    if separation.distance > tol:
        return None
    if "sphere" in (shape(a), shape(b)):
        return _single_point(separation)
    if isinstance(b, Plane):
        assert isinstance(a, Solid)
        return _plane_manifold(a, b, tol)
    if isinstance(a, Plane):
        return _flip_manifold(_plane_manifold(b, a, tol))
    return _clipped_manifold(polyhedron(a), polyhedron(b), separation, tol)


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
    vertices = polyhedron(a).vertices
    heights = vertices @ b.normal - b.offset
    near = heights <= max(tol, float(heights.min()) + 1e-9)
    return _patch(vertices[near], heights[near], b.normal)


def _clipped_manifold(a: Polyhedron, b: Polyhedron, sep: Separation, tol: float) -> Manifold:
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
        if (d_prev <= EPS) != (d_cur <= EPS) and count > 1:
            kept.append(previous + (current - previous) * (d_prev / (d_prev - d_cur)))
        if d_cur <= EPS:
            kept.append(current)
    return np.array(kept).reshape(-1, 3)


def convex_hull_indices(points_2d: ArrayLike) -> NDArray[np.int64]:
    """Counter-clockwise hull vertex indices (monotone chain, collinear points dropped)."""
    xy = np.asarray(points_2d, dtype=np.float64)
    span = float(np.ptp(xy, axis=0).max()) if xy.shape[0] else 0.0
    if span < EPS:
        return np.zeros(min(1, xy.shape[0]), dtype=np.int64)
    tol = EPS * span * span
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
