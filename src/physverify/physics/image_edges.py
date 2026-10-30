"""Rows where an object's projected box touches the image border: its mask is cut there, so its
extents describe the visible part, not the object."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physverify.schema.perception import CameraModel
from physverify.schema.states import CoordFrame, ObjectStateTable

BoolArray = NDArray[np.bool_]


def near_image_edge(states: ObjectStateTable, camera: CameraModel, margin_px: float) -> BoolArray:
    """(T, N) True where the box of half-widths f e / (2 z) around the projected centre
    (u = fx x / z + cx, v = fy y / z + cy) lies within ``margin_px`` of a border of the
    processing image. All False for a table not in CAMERA_OPENCV (its extents are not image
    aligned); unknown positions or extents never flag a row."""
    shape = states.position.shape[:2]
    if states.frame is not CoordFrame.CAMERA_OPENCV:
        return np.zeros(shape, dtype=np.bool_)
    k = camera.K
    x, y, z = (states.position[..., axis] for axis in range(3))
    with np.errstate(invalid="ignore", divide="ignore"):
        u, v = k[0, 0] * x / z + k[0, 2], k[1, 1] * y / z + k[1, 2]
        half_u = k[0, 0] * states.size[..., 0] / (2.0 * z)
        half_v = k[1, 1] * states.size[..., 1] / (2.0 * z)
        width, height = camera.sampling.width, camera.sampling.height
        near = (
            (u - half_u < margin_px)
            | (u + half_u > width - margin_px)
            | (v - half_v < margin_px)
            | (v + half_v > height - margin_px)
        )
    return np.asarray(near, dtype=np.bool_)
