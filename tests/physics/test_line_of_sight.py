import numpy as np
import pytest

from physverify.physics.line_of_sight import line_of_sight_covariance
from physverify.schema.states import CoordFrame

CAMERA = CoordFrame.CAMERA_OPENCV
FLOOR = 1e-6


def rows(n: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    position = np.column_stack([rng.uniform(-0.5, 0.5, (n, 2)), rng.uniform(1.0, 4.0, n)])
    rel = rng.uniform(0.01, 0.05, n)
    lateral = [np.hypot(rel * position[:, k], 0.003) for k in (0, 1)]
    sigma = np.column_stack([*lateral, rel * position[:, 2]])
    return position, sigma


def covariance(position: np.ndarray, sigma: np.ndarray, frame: CoordFrame = CAMERA) -> np.ndarray:
    return line_of_sight_covariance(position, sigma, sigma == 0.0, frame, FLOOR)


def test_the_diagonal_is_the_per_axis_variance_up_to_the_floor() -> None:
    position, sigma = rows()
    cov = covariance(position, sigma)
    diagonal = np.diagonal(cov, axis1=1, axis2=2)
    np.testing.assert_allclose(diagonal, sigma**2, rtol=1e-12, atol=2 * FLOOR**2)


def test_the_rest_of_the_diagonal_is_a_rank_one_term_along_the_ray() -> None:
    position, sigma = rows()
    cov = covariance(position, sigma)
    rel2 = (sigma[:, 2] / position[:, 2]) ** 2
    rest = np.maximum(sigma**2 - rel2[:, None] * position**2, FLOOR**2)
    ray = cov - np.einsum("nk,kl->nkl", rest, np.eye(3))
    expected = rel2[:, None, None] * np.einsum("nk,nl->nkl", position, position)
    np.testing.assert_allclose(ray, expected, rtol=1e-9, atol=1e-15)
    assert [np.linalg.matrix_rank(r, tol=1e-12) for r in ray] == [1] * len(ray)


def test_every_row_is_symmetric_positive_definite() -> None:
    position, sigma = rows(40, seed=3)
    cov = covariance(position, sigma)
    np.testing.assert_array_equal(cov, np.swapaxes(cov, 1, 2))
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)


def test_outside_the_camera_frame_it_is_the_diagonal() -> None:
    position, sigma = rows()
    cov = covariance(position, sigma, CoordFrame.WORLD_ZUP)
    np.testing.assert_array_equal(cov, np.einsum("nk,kl->nkl", sigma**2, np.eye(3)))


def test_a_row_behind_the_camera_keeps_the_diagonal() -> None:
    position, sigma = rows()
    position[2, 2] = -1.0
    cov = covariance(position, sigma)
    np.testing.assert_array_equal(cov[2], np.diag(sigma[2] ** 2))


def test_exact_rows_take_the_floor_on_every_axis() -> None:
    position, sigma = rows()
    sigma[1] = 0.0
    cov = covariance(position, sigma)
    np.testing.assert_array_equal(cov[1], FLOOR**2 * np.eye(3))


def test_a_rest_below_the_floor_is_raised_to_it() -> None:
    position = np.array([[0.0, 0.0, 2.0]])
    sigma = np.array([[0.0, 0.0, 0.1]])
    cov = line_of_sight_covariance(position, sigma, np.zeros((1, 3), bool), CAMERA, 1e-3)
    assert cov[0, 0, 0] == pytest.approx(1e-6)
    assert cov[0, 1, 1] == pytest.approx(1e-6)
    assert cov[0, 2, 2] == pytest.approx(0.01 + 1e-6)
