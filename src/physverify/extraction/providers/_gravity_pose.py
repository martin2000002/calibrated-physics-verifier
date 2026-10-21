"""Gravity direction <-> camera pose, in OpenCV camera axes (x right, y down, z forward).

The world is z-up (Kubric), so world down is (0, 0, -1) and a camera pose ``world_from_camera``
gives the gravity in camera axes as R_wc^T (0, 0, -1). A level camera sees gravity along +y.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

WORLD_DOWN = np.array([0.0, 0.0, -1.0])
LEVEL_CAMERA_GRAVITY = np.array([0.0, 1.0, 0.0])
CAMERA_RIGHT = np.array([1.0, 0.0, 0.0])
CAMERA_FORWARD = np.array([0.0, 0.0, 1.0])
DEGENERATE_DOT = 0.9


def pose_rotation(world_from_camera: FloatArray) -> FloatArray:
    """R_wc of a (4, 4) pose, or of the first row of a (T, 4, 4) one."""
    pose = np.asarray(world_from_camera, dtype=np.float64)
    rotation: FloatArray = pose[0, :3, :3] if pose.ndim == 3 else pose[:3, :3]
    return rotation


def gravity_from_pose(world_from_camera: FloatArray) -> FloatArray:
    """Gravity in camera axes: R_wc^T (0, 0, -1)."""
    gravity: FloatArray = pose_rotation(world_from_camera).T @ WORLD_DOWN
    return gravity


def pose_from_gravity(gravity: FloatArray) -> FloatArray:
    """A (4, 4) pose with translation 0 whose gravity is ``gravity`` (normalised).

    Row 3 of R_wc is -g; row 1 is the camera x axis made orthogonal to it (the camera z axis when
    x is nearly parallel to g) and row 2 completes a right-handed basis. The yaw this fixes and
    the zero translation are conventions, not measurements."""
    g = np.asarray(gravity, dtype=np.float64)
    up = -g / np.linalg.norm(g)
    helper = CAMERA_RIGHT if abs(float(CAMERA_RIGHT @ up)) < DEGENERATE_DOT else CAMERA_FORWARD
    east = helper - (helper @ up) * up
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    pose = np.eye(4)
    pose[:3, :3] = np.vstack([east, north, up])
    return pose
