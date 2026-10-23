"""R2 support: overhang of the centre of mass beyond the support polygon [overhang_over_size]."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physverify.physics.geometry import (
    characteristic_length,
    convex_hull_indices,
    cross_2d,
    project_to_plane,
)


def overhang_fraction(
    com: ArrayLike, support_polygon: ArrayLike, gravity_dir: ArrayLike, extents: ArrayLike
) -> float:
    """Signed horizontal distance from the centre of mass to the support polygon boundary,
    positive outside and negative inside, divided by the characteristic length L.

    Everything is projected on the plane orthogonal to ``gravity_dir``. A polygon that
    projects to a segment or a point has no inside, so the result is >= 0.
    """
    points = np.asarray(support_polygon, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] == 0:
        raise ValueError("support polygon needs at least one point")
    xy = project_to_plane(points, gravity_dir)
    target = project_to_plane(com, gravity_dir)[0]
    hull = xy[convex_hull_indices(xy)]
    return _signed_distance_2d(target, hull) / characteristic_length(extents)


def _signed_distance_2d(point: NDArray[np.float64], hull: NDArray[np.float64]) -> float:
    if hull.shape[0] == 1:
        return float(np.linalg.norm(point - hull[0]))
    edges = [(hull[i], hull[(i + 1) % hull.shape[0]]) for i in range(hull.shape[0])]
    distance = min(_segment_distance(point, a, b) for a, b in edges)
    if hull.shape[0] < 3:
        return distance
    inside = all(cross_2d(a, b, point) >= 0 for a, b in edges)
    return -distance if inside else distance


def _segment_distance(
    point: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]
) -> float:
    edge = b - a
    fraction = float(np.clip((point - a) @ edge / (edge @ edge), 0.0, 1.0))
    return float(np.linalg.norm(point - (a + fraction * edge)))
