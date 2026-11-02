"""R4_jump rigid2, shared by its variants: v2 without the stencils where the object's size steps,
in the statistic and in its noise estimate."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks._base import CheckInputs
from physverify.checks.r4_track_jump import NoiseContinuityCheck, NoiseJumpParams
from physverify.physics.size_steps import size_step_stencil_mask
from physverify.schema.contacts import ContactTable

RIGID_SIZE_STEP = 0.10
"""A rigid body's size does not step between rows; a 10 % step is a mask or depth change."""


class RigidJumpParams(NoiseJumpParams):
    rigid_size_step: float = Field(default=RIGID_SIZE_STEP, gt=0.0)


class RigidJumpCheck(NoiseContinuityCheck):
    """R4_jump rigid2 [jump_over_noise]: the v2 statistic and its noise estimate (``track_noise``)
    over the rows whose stencil spans neither an impact nor a step |L_{j+1} / L_j - 1| >
    ``rigid_size_step`` of the object's length L = (e_x e_y e_z)^(1/3)
    (``size_step_stencil_mask``).

    The extents of a perceived table are the mask scaled by its depth, so a mask that swaps to
    another region or loses part of the body steps L and moves the centroid with it; a body
    displaced in depth keeps L. Such a step is a perception error, not position noise, so it is
    left out of sigma-hat as well ("rigid" kept it there)."""

    variant: ClassVar[str] = "rigid2"
    version: ClassVar[str] = "1"
    Params: ClassVar[type[BaseModel]] = RigidJumpParams
    params: RigidJumpParams

    def __init__(self, params: BaseModel) -> None:
        super().__init__(RigidJumpParams.model_validate(params))

    def _excluded(
        self, inputs: CheckInputs, contacts: ContactTable, index: int
    ) -> NDArray[np.bool_]:
        steps = size_step_stencil_mask(inputs.states.size[:, index], self.params.rigid_size_step)
        return super()._excluded(inputs, contacts, index) | steps
