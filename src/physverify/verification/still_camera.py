"""The still-camera precondition: a check that measures in the camera frame abstains with
``CAMERA_NOT_STILL`` when the clip's background moves.

It reads only the clip's ``CameraMotion`` (pixels and TRACK masks) and the check's declaration,
never a magnitude, verdict or label. A clip without a measured background is never gated."""

from __future__ import annotations

import math
from typing import Any

from pydantic import Field, field_validator

from physverify.checks._base import Check, CheckInputs, abstain, assumes_still_camera
from physverify.core.models import FrozenModel
from physverify.schema.checks import AbstainReason, CheckResult

STILL_CAMERA_KEY = "still_camera"
"""``[verification.still_camera]``; without it there is no gate."""


class StillCameraSettings(FrozenModel):
    """``estimator`` and its ``params`` measure the motion; a clip is gated when the 90th
    percentile over its pairs exceeds ``p90_limit`` or, when set, its maximum exceeds
    ``max_limit``, in diagonals per pair or, with ``per_second``, per second."""

    estimator: str = "affine_flow"
    params: dict[str, Any] = Field(default_factory=dict)
    per_second: bool
    p90_limit: float = Field(gt=0.0)
    max_limit: float | None = Field(default=None, gt=0.0)

    @field_validator("p90_limit", "max_limit")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("a limit must be finite")
        return value


def camera_gate(
    check: Check, inputs: CheckInputs, settings: StillCameraSettings
) -> CheckResult | None:
    """The ``CAMERA_NOT_STILL`` abstention of ``check`` on this clip, or None when the check does
    not assume a still camera, the clip has no measured motion or the motion is within limits."""
    motion = inputs.camera_motion
    if not assumes_still_camera(check) or motion is None:
        return None
    statistics = motion.clip_statistics(settings.per_second)
    if statistics is None:
        return None
    p90, peak = statistics
    over_max = settings.max_limit is not None and peak > settings.max_limit
    if p90 <= settings.p90_limit and not over_max:
        return None
    details = {
        "camera_p90": p90,
        "camera_peak": peak,
        "p90_limit": settings.p90_limit,
        "max_limit": settings.max_limit,
        "per_second": settings.per_second,
    }
    return abstain(type(check), inputs, (), AbstainReason.CAMERA_NOT_STILL, None, details)
