"""R4_jump rigid2_dfloor: rigid2 with the depth axis of sigma-hat floored at the depth sigma the
lift declares."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from physverify.checks._base import CheckInputs
from physverify.checks.variants._rigid_jump import RigidJumpCheck
from physverify.schema.states import CoordFrame, ObjectStateTable

DEPTH_AXIS = 2
"""z of CAMERA_OPENCV, the axis along which a monocular lift declares its depth sigma."""


def declared_depth_sigma(
    states: ObjectStateTable, index: int, excluded: NDArray[np.bool_]
) -> float:
    """Median declared depth sigma of object ``index`` over the rows not ``excluded``; NaN in
    another frame or without a finite one."""
    if states.frame is not CoordFrame.CAMERA_OPENCV:
        return math.nan
    sigma = states.position_sigma[~excluded, index, DEPTH_AXIS]
    known = sigma[np.isfinite(sigma)]
    return float(np.median(known)) if known.size else math.nan


class DeclaredDepthJumpCheck(RigidJumpCheck):
    """R4_jump rigid2_dfloor [jump_over_noise]: rigid2 with sigma-hat on the camera depth axis no
    smaller than the median depth sigma the lift declares per row (``declared_depth_sigma``); x
    and y keep the pixel floor, and exact positions (declared sigma 0) or an unknown sigma keep
    rigid2's floor.

    The third differences of the track do not see the per-row depth noise of a monocular lift
    (about 0.39 z on calibration), so v2 reads it as jumps. The price is a declared blind spot:
    a jump along the line of sight smaller than about sqrt(20) times the declared depth sigma
    (times tau) is not seen, since per-row monocular depth cannot support it."""

    variant: ClassVar[str] = "rigid2_dfloor"
    version: ClassVar[str] = "1"

    def _floor(
        self, inputs: CheckInputs, index: int, excluded: NDArray[np.bool_], pixel: float
    ) -> NDArray[np.float64]:
        floor = super()._floor(inputs, index, excluded, pixel)
        depth = declared_depth_sigma(inputs.states, index, excluded)
        if math.isfinite(depth):
            floor[DEPTH_AXIS] = max(floor[DEPTH_AXIS], depth)
        return floor


PLUGINS = (DeclaredDepthJumpCheck,)
