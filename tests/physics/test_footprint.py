"""Outline points of boxes, cylinders and spheres under rotation, and who rests on whom."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.physics.footprint import outline_points, support_order
from physverify.physics.geometry import quaternion_from_matrix

UP = np.array([0.0, 0.0, 1.0])


def about_z(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return quaternion_from_matrix(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))


def about_x(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return quaternion_from_matrix(np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]))


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_a_box_gives_its_eight_corners() -> None:
    points = outline_points((1.0, 2.0, 3.0), (0.4, 0.2, 0.1), IDENTITY, "box")
    assert points.shape == (8, 3)
    np.testing.assert_allclose(points.min(axis=0), (0.8, 1.9, 2.95))
    np.testing.assert_allclose(points.max(axis=0), (1.2, 2.1, 3.05))


def test_a_box_turned_45_degrees_reaches_its_half_diagonal() -> None:
    points = outline_points((0.0, 0.0, 0.0), (0.2, 0.2, 0.2), about_z(45.0), "box")
    np.testing.assert_allclose(np.abs(points[:, :2]).max(axis=0), math.sqrt(2) * 0.1, atol=1e-12)
    np.testing.assert_allclose(np.abs(points[:, 2]), 0.1, atol=1e-12)


def test_an_unknown_primitive_is_a_box() -> None:
    box = outline_points((0.0, 0.0, 0.0), (0.4, 0.2, 0.1), about_z(30.0), "box")
    unknown = outline_points((0.0, 0.0, 0.0), (0.4, 0.2, 0.1), about_z(30.0), "unknown")
    np.testing.assert_allclose(unknown, box)


def test_an_upright_cylinder_has_two_rims_of_its_radius() -> None:
    points = outline_points((0.0, 0.0, 1.0), (0.2, 0.2, 0.6), IDENTITY, "cylinder", n_rim=12)
    assert points.shape == (24, 3)
    np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), 0.1, atol=1e-12)
    np.testing.assert_allclose(sorted(set(np.round(points[:, 2], 12))), (0.7, 1.3))


def test_a_cylinder_on_its_side_rolls_its_rims_into_the_vertical() -> None:
    points = outline_points((0.0, 0.0, 0.0), (0.2, 0.2, 0.6), about_x(90.0), "cylinder")
    np.testing.assert_allclose(np.abs(points[:, 1]), 0.3, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(points[:, [0, 2]], axis=1), 0.1, atol=1e-12)


def test_a_sphere_is_its_centre() -> None:
    points = outline_points((1.0, 2.0, 3.0), (0.2, 0.2, 0.2), IDENTITY, "sphere")
    np.testing.assert_allclose(points, [[1.0, 2.0, 3.0]])


@pytest.mark.parametrize("bad", [(np.nan, 0.0, 0.0), (0.0, np.inf, 0.0)])
def test_a_non_finite_centre_is_refused(bad: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        outline_points(bad, (0.2, 0.2, 0.2), IDENTITY, "box")


def test_a_rim_needs_three_points() -> None:
    with pytest.raises(ValueError, match="3 points"):
        outline_points((0.0, 0.0, 0.0), (0.2, 0.2, 0.2), IDENTITY, "cylinder", n_rim=2)


def test_the_normal_from_b_to_a_puts_a_on_top_either_way() -> None:
    assert support_order("a", "b", UP, UP, 45.0) == ("a", "b")
    assert support_order("a", "b", -UP, UP, 45.0) == ("b", "a")


@pytest.mark.parametrize(("tilt", "expected"), [(40.0, ("a", "b")), (60.0, None), (90.0, None)])
def test_a_normal_steeper_than_the_limit_is_no_support(
    tilt: float, expected: tuple[str, str] | None
) -> None:
    t = math.radians(tilt)
    normal = np.array([math.sin(t), 0.0, math.cos(t)])
    assert support_order("a", "b", normal, UP, 45.0) == expected
