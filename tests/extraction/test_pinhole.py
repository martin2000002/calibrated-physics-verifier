"""Pinhole back-projection: the inverse of the projection u = f_x X / Z + c_x."""

from __future__ import annotations

import numpy as np

from physverify.extraction.pinhole import ray_points

K = np.array([[60.0, 0.0, 15.5], [0.0, 58.0, 12.0], [0.0, 0.0, 1.0]])


def test_the_principal_point_lies_on_the_optical_axis() -> None:
    points = ray_points(np.array([15.5]), np.array([12.0]), np.array([3.0]), K)
    np.testing.assert_array_equal(points, [[0.0, 0.0, 3.0]])


def test_projecting_the_points_gives_back_their_pixels() -> None:
    rng = np.random.default_rng(0)
    u, v, z = rng.uniform(0, 32, 50), rng.uniform(0, 24, 50), rng.uniform(0.5, 5.0, 50)
    points = ray_points(u, v, z, K)
    assert points.shape == (50, 3)
    np.testing.assert_allclose(points[:, 0] / points[:, 2] * K[0, 0] + K[0, 2], u, atol=1e-12)
    np.testing.assert_allclose(points[:, 1] / points[:, 2] * K[1, 1] + K[1, 2], v, atol=1e-12)
    np.testing.assert_array_equal(points[:, 2], z)
