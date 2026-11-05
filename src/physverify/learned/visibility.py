"""Positives that are invisible by construction, for the invisible-positive control.

- Interpenetration: the ground-truth contact zone of the perturbed pair never shows a pixel
  (its ``zone_px_share_max`` over the clip is 0).
- Support offset: the recorded offset points within ``max_angle_deg`` of the camera's line of
  sight, so the overhang barely moves the image. The line of sight is approximated by the ray
  from the camera position to its look-at point, not to the moved body.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from physverify.schema.scene import CameraSpec, PerturbationRecord

OFFSET_KEY = "offset_m"
MAX_ANGLE_DEG = 30.0


def zone_invisible(zone_share_max: float) -> bool:
    return zone_share_max == 0.0


def ray_angle_deg(offset: Sequence[float], camera: CameraSpec) -> float:
    """Angle between ``offset`` (either sign) and the camera ray; NaN for a zero vector."""
    ray = np.subtract(camera.look_at, camera.position)
    vec = np.asarray(offset, dtype=np.float64)
    norms = float(np.linalg.norm(ray) * np.linalg.norm(vec))
    if norms == 0.0:
        return math.nan
    cosine = min(abs(float(ray @ vec)) / norms, 1.0)
    return math.degrees(math.acos(cosine))


def offset_invisible(
    record: PerturbationRecord, camera: CameraSpec, max_angle_deg: float = MAX_ANGLE_DEG
) -> bool:
    offset = record.params.get(OFFSET_KEY)
    if not isinstance(offset, list | tuple):
        return False
    angle = ray_angle_deg([float(v) for v in offset], camera)
    return bool(angle <= max_angle_deg)
