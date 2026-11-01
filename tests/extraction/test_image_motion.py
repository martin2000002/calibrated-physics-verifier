"""Image direction of a 3D vector at an object: the derivative of the pinhole projection."""

from __future__ import annotations

import numpy as np
import pytest
import scene_contact_cases as cases

from physverify.extraction.providers._image_motion import ImageProjection

cs = cases.cs
FOCAL = np.array([500.0, 500.0])
ROWS = 5


def projection(position: tuple[float, float, float], rotation: np.ndarray) -> ImageProjection:
    positions = np.tile(np.asarray(position, dtype=np.float64), (ROWS, 1, 1))
    return ImageProjection(positions, np.tile(rotation, (ROWS, 1, 1)), FOCAL)


def test_camera_up_at_the_optical_axis_points_to_the_top_of_the_image() -> None:
    found = projection((0.0, 0.0, 4.0), np.eye(3)).direction(0, 2, np.array([0.0, -1.0, 0.0]))
    assert found == pytest.approx([0.0, -1.0])


def test_a_receding_object_off_axis_moves_towards_the_centre() -> None:
    found = projection((1.0, 0.0, 4.0), np.eye(3)).direction(0, 2, np.array([0.0, 0.0, 1.0]))
    assert found == pytest.approx([-1.0, 0.0])


def test_a_world_vector_is_rotated_into_the_camera_first() -> None:
    camera_from_world = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    found = projection((0.0, 0.0, 4.0), camera_from_world).direction(0, 2, np.array([0, 0, 1.0]))
    assert found == pytest.approx([0.0, -1.0])


def test_the_scene_up_matches_the_projected_displacement_of_the_object() -> None:
    case = cases.boxes_at([0.0, 0.3])
    up = case.view.vector(cs.sc.UP)
    image = ImageProjection.of(case.states, cs.scene_camera(case))
    for n in range(2):
        point = case.states.position[3, n]
        moved = point + 1e-6 * up
        shift = cs.FOCAL_PX * (moved[:2] / moved[2] - point[:2] / point[2])
        found = image.direction(n, 3, up)
        assert found == pytest.approx(shift / np.linalg.norm(shift), abs=1e-6)


def test_rows_behind_the_camera_are_skipped_and_none_ahead_gives_none() -> None:
    image = projection((1.0, 0.0, 4.0), np.eye(3))
    assert image.positions is not None
    image.positions[2:, 0, 2] = -4.0
    assert image.direction(0, 4, np.array([0.0, 0.0, 1.0])) == pytest.approx([-1.0, 0.0])
    image.positions[:, 0, 2] = -4.0
    assert image.direction(0, 4, np.array([0.0, 0.0, 1.0])) is None


def test_without_a_pose_that_fits_the_rows_there_is_no_direction() -> None:
    image = ImageProjection(None, None, FOCAL)
    assert image.direction(0, 0, np.array([0.0, 0.0, 1.0])) is None
