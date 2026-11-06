"""Positives that are invisible by construction, for the invisible-positive control.

- Interpenetration: the ground-truth contact zone of the perturbed pair never shows a pixel
  (its ``zone_px_share_max`` over the clip is 0).
- Support offset: the recorded offset points strictly within ``max_angle_deg`` of the camera's
  line of sight, so the overhang barely moves the image. The line of sight is approximated by the
  ray from the camera position to its look-at point, not to the moved body.

``invisible`` answers per check; None when the evidence it needs is missing (no zone share, no
camera, no recorded offset): the positive is then not marked invisible.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

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
) -> bool | None:
    """None when the record holds no offset."""
    offset = record.params.get(OFFSET_KEY)
    if not isinstance(offset, list | tuple):
        return None
    angle = ray_angle_deg([float(v) for v in offset], camera)
    return bool(angle < max_angle_deg)


@dataclass(frozen=True)
class PositiveEvidence:
    """``zone_max``: the ground-truth zone share of the target subject; ``camera``: the scene's."""

    record: PerturbationRecord
    camera: CameraSpec | None
    zone_max: float | None


def _zone_rule(evidence: PositiveEvidence) -> bool | None:
    zone = evidence.zone_max
    return None if zone is None or math.isnan(zone) else zone_invisible(zone)


def _offset_rule(evidence: PositiveEvidence) -> bool | None:
    camera = evidence.camera
    return None if camera is None else offset_invisible(evidence.record, camera)


INVISIBLE_RULES: Mapping[str, Callable[[PositiveEvidence], bool | None]] = {
    "R1": _zone_rule,
    "R2": _offset_rule,
}


def invisible(check_id: str, evidence: PositiveEvidence) -> bool | None:
    rule = INVISIBLE_RULES.get(check_id)
    return None if rule is None else rule(evidence)
