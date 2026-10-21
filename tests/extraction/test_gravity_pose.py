"""it2 T1-T2: gravity <-> pose in one place, and LIFT's gravity unchanged by the refactor."""

from __future__ import annotations

import math

import numpy as np
import pytest
from support import make_context
from synthetic_clip import SyntheticClip, world_from_camera

from physverify.extraction.providers._gravity_pose import (
    LEVEL_CAMERA_GRAVITY,
    gravity_from_pose,
    pose_from_gravity,
)
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.schema.perception import CameraModel, IntrinsicsSource

TOL = 1e-12
DIRECTIONS = 50


def directions() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    nearly_along_x = np.array([-0.95, 0.1, 0.3])
    vectors = [*rng.normal(size=(DIRECTIONS, 3)), np.array([1.0, 0.0, 0.0]), nearly_along_x]
    return [v / np.linalg.norm(v) for v in vectors]


@pytest.mark.parametrize("gravity", directions())
def test_t1_pose_from_gravity_is_a_rotation_that_gives_the_gravity_back(
    gravity: np.ndarray,
) -> None:
    pose = pose_from_gravity(3.0 * gravity)
    rotation = pose[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=TOL)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=TOL)
    np.testing.assert_array_equal(pose[:3, 3], np.zeros(3))
    np.testing.assert_array_equal(pose[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(gravity_from_pose(pose), gravity, atol=TOL)


def test_a_level_camera_pose_has_the_level_gravity() -> None:
    np.testing.assert_allclose(
        gravity_from_pose(pose_from_gravity(LEVEL_CAMERA_GRAVITY)), LEVEL_CAMERA_GRAVITY, atol=TOL
    )


def old_gravity(pose: np.ndarray) -> np.ndarray:
    """LIFT version 2's inline formula: R_wc^T (0, 0, -1) of the (first) pose."""
    rotation = pose[0, :3, :3] if pose.ndim == 3 else pose[:3, :3]
    out: np.ndarray = rotation.T @ np.array([0.0, 0.0, -1.0])
    return out


@pytest.mark.parametrize("per_frame", [False, True])
def test_t2_lift_reads_the_same_gravity_as_before(
    fixture_clip: SyntheticClip, per_frame: bool
) -> None:
    sampling = make_context(fixture_clip).sampling
    pose = world_from_camera()
    if per_frame:
        pose = np.repeat(pose[None], sampling.n_frames, axis=0)
    camera = CameraModel(
        sampling=sampling,
        K=np.eye(3),
        source=IntrinsicsSource.ESTIMATED,
        fov_deg=60.0,
        fov_sigma_deg=10.0,
        world_from_camera=pose,
    )
    lift = BackprojectLift(BackprojectLift.Params())
    gravity, sigma = lift._gravity(camera, "auto")  # noqa: SLF001
    np.testing.assert_array_equal(gravity, old_gravity(pose))
    assert sigma == pytest.approx(math.radians(2.0))
    level, level_sigma = lift._gravity(camera, "level_camera")  # noqa: SLF001
    np.testing.assert_array_equal(level, LEVEL_CAMERA_GRAVITY)
    assert level_sigma == pytest.approx(math.radians(10.0))
