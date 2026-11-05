from __future__ import annotations

import numpy as np

from physverify.checks._image import depth_jacobian, project, row_sigma

K = np.array([[300.0, 0.0, 128.0], [0.0, 310.0, 127.0], [0.0, 0.0, 1.0]])


def numeric_jacobian(point: np.ndarray, step: float = 1e-6) -> np.ndarray:
    columns = [
        (project(K, point + step * axis) - project(K, point - step * axis)) / (2.0 * step)
        for axis in np.eye(3)
    ]
    return np.stack(columns, axis=1)


def test_the_depth_jacobian_over_z_is_the_jacobian_of_the_projection() -> None:
    for point in ([0.3, -0.2, 2.0], [-1.0, 0.5, 6.0], [0.0, 0.0, 1.0]):
        p = np.array(point)
        found = depth_jacobian(K, project(K, p)) / p[2]
        np.testing.assert_allclose(found, numeric_jacobian(p), rtol=1e-6, atol=1e-6)


def test_the_depth_jacobian_broadcasts_over_pixels() -> None:
    points = np.array([[0.3, -0.2, 2.0], [-1.0, 0.5, 6.0]])
    batched = depth_jacobian(K, project(K, points))
    for row, point in zip(batched, points, strict=True):
        np.testing.assert_array_equal(row, depth_jacobian(K, project(K, point)))


def test_the_row_sigma_is_the_row_jacobian_weighted_by_the_axis_sigma() -> None:
    point, sigma = np.array([0.3, -0.2, 2.0]), np.array([0.01, 0.02, 0.05])
    expected = float(np.sqrt(np.sum((numeric_jacobian(point)[1] * sigma) ** 2)))
    assert np.isclose(row_sigma(K, point, sigma), expected, rtol=1e-6)
