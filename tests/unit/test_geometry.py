from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.physics.geometry import (
    CYLINDER_PRISM_REL_ERROR,
    Plane,
    Pose,
    Primitive,
    Solid,
    contact_manifold,
    matrix_from_quaternion,
    polygon_area,
    quaternion_from_matrix,
    reduce_manifold,
    reduce_manifold_indices,
    signed_distance,
    support_polygon,
)
from physverify.schema.states import PrimitiveKind

UP = np.array([0.0, 0.0, 1.0])


def solid(
    kind: PrimitiveKind,
    extents: tuple[float, float, float],
    position: tuple[float, float, float],
    rotation: np.ndarray | None = None,
) -> Solid:
    pose = Pose(np.array(position, dtype=float), np.eye(3) if rotation is None else rotation)
    return Solid(Primitive(kind, np.array(extents, dtype=float)), pose)


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_sphere_sphere() -> None:
    a = solid(PrimitiveKind.SPHERE, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    b = solid(PrimitiveKind.SPHERE, (0.6, 0.6, 0.6), (1.2, 0.9, 0.0))
    result = signed_distance(a, b)
    assert abs(result.distance - (1.5 - 0.5 - 0.3)) < 1e-12
    np.testing.assert_allclose(result.normal, [-0.8, -0.6, 0.0], atol=1e-12)


def test_sphere_plane_penetration() -> None:
    ball = solid(PrimitiveKind.SPHERE, (1.0, 1.0, 1.0), (0.3, -2.0, 0.3))
    result = signed_distance(ball, Plane.ground())
    assert abs(result.distance + 0.2) < 1e-12
    np.testing.assert_allclose(result.normal, UP, atol=1e-12)


def test_box_on_plane_has_four_point_manifold() -> None:
    box = solid(PrimitiveKind.BOX, (1.0, 2.0, 0.5), (3.0, 1.0, 0.25))
    manifold = contact_manifold(box, Plane.ground(), tol=1e-6)
    assert manifold is not None and manifold.points.shape == (4, 3)
    np.testing.assert_allclose(manifold.points[:, 2], 0.0, atol=1e-12)
    assert abs(polygon_area(manifold.points, UP) - 2.0) < 1e-9


def test_separated_bodies_have_no_manifold() -> None:
    box = solid(PrimitiveKind.BOX, (1.0, 1.0, 1.0), (0.0, 0.0, 0.6))
    assert contact_manifold(box, Plane.ground(), tol=0.05) is None


def test_box_on_offset_box_gives_intersection_rectangle() -> None:
    lower = solid(PrimitiveKind.BOX, (1.0, 1.0, 1.0), (0.0, 0.0, 0.5))
    upper = solid(PrimitiveKind.BOX, (1.0, 1.0, 1.0), (0.3, 0.2, 1.5))
    assert abs(signed_distance(upper, lower).distance) < 1e-12
    manifold = contact_manifold(upper, lower, tol=1e-6)
    assert manifold is not None
    assert abs(polygon_area(manifold.points, UP) - 0.7 * 0.8) < 1e-9
    np.testing.assert_allclose(manifold.points[:, 2], 1.0, atol=1e-12)
    xs, ys = manifold.points[:, 0], manifold.points[:, 1]
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == pytest.approx((-0.2, 0.5, -0.3, 0.5))


def test_tilted_cylinder_plane_is_exact() -> None:
    radius, height, tilt = 0.3, 1.0, 0.4
    cylinder = solid(
        PrimitiveKind.CYLINDER, (2 * radius, 2 * radius, height), (0.0, 0.0, 2.0), rotation_x(tilt)
    )
    lowest = 2.0 - (height / 2 * math.cos(tilt) + radius * math.sin(tilt))
    assert abs(signed_distance(cylinder, Plane.ground()).distance - lowest) < 1e-12


def test_cylinder_on_its_side_against_box_within_prism_bound() -> None:
    radius = 0.25
    lying = rotation_x(math.pi / 2)
    for roll in np.linspace(0.0, math.pi / 8, 5):
        rotation = lying @ np.array(
            [
                [math.cos(roll), -math.sin(roll), 0.0],
                [math.sin(roll), math.cos(roll), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        cylinder = solid(
            PrimitiveKind.CYLINDER, (2 * radius, 2 * radius, 1.0), (0.0, 0.0, 1.3), rotation
        )
        table = solid(PrimitiveKind.BOX, (2.0, 2.0, 1.0), (0.0, 0.0, 0.5))
        error = abs(signed_distance(cylinder, table).distance - (1.3 - radius - 1.0))
        assert error <= CYLINDER_PRISM_REL_ERROR * radius + 1e-12


def test_sphere_box_edge_distance() -> None:
    box = solid(PrimitiveKind.BOX, (2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
    ball = solid(PrimitiveKind.SPHERE, (0.5, 0.5, 0.5), (2.0, 2.0, 0.0))
    assert abs(signed_distance(ball, box).distance - (math.sqrt(2) - 0.25)) < 1e-12


def test_reduce_manifold_keeps_at_most_eight_hull_points() -> None:
    rng = np.random.default_rng(3)
    points = np.column_stack([rng.normal(size=(60, 2)), np.zeros(60)])
    reduced = reduce_manifold(points, UP)
    indices = reduce_manifold_indices(points, UP)
    assert 3 <= reduced.shape[0] <= 8
    np.testing.assert_array_equal(reduced, points[indices])
    assert reduce_manifold(points[:3], UP).shape[0] == 3


def test_support_polygon_of_resting_box() -> None:
    box = solid(PrimitiveKind.BOX, (1.0, 1.0, 1.0), (0.0, 0.0, 0.5))
    manifold = contact_manifold(box, Plane.ground(), tol=1e-6)
    assert manifold is not None
    polygon = support_polygon([manifold], -UP)
    assert abs(polygon_area(polygon, UP) - 1.0) < 1e-9


def test_quaternion_round_trip() -> None:
    q = np.array([0.8, 0.2, -0.4, 0.4])
    q /= np.linalg.norm(q)
    rotation = matrix_from_quaternion(q)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    back = quaternion_from_matrix(rotation)
    assert min(np.abs(back - q).max(), np.abs(back + q).max()) < 1e-12
