"""R4_jump rigid2_dfloor_shape: rigid2_dfloor_null that also abstains when the peak is not shaped
like a displacement of the body."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks.variants._null_tail_jump import NullTailJumpCheck, NullTailJumpParams
from physverify.physics.measures.jump_shape import fit_shapes

SHAPE_LEVEL = 6.0
"""Peak (raw jump_over_noise) from which its shape is read: on calibration the shapes of lower
peaks were those of noise (teleports and clean tracks alike), the shapes above it told a teleport
(24 of 26 steps) from a clean peak (2 of 10). Below it, PASS and the ranking are those of
rigid2_dfloor_null."""


class StepShapeJumpParams(NullTailJumpParams):
    shape_level: float = Field(default=SHAPE_LEVEL, ge=0.0)


class StepShapeJumpCheck(NullTailJumpCheck):
    """R4_jump rigid2_dfloor_shape [jump_over_noise]: rigid2_dfloor_null, but ABSTAIN
    (UNKNOWN_UNCERTAINTY, ``shape`` in details) when the third differences around the peak are
    fitted better by a kink (the velocity changes at one frame: an impact the contact table does
    not hold, as on real video with no contact records) or by a one-frame outlier (a repeated or
    late frame, a mask that swaps region for a frame) than by a step (``fit_shapes``), or when the
    rows around the peak cannot tell a step from them; only for a peak of at least
    ``shape_level``.

    A teleport moves the body and leaves it there, a step of the positions; the other two are
    shapes the statistic also answers to but that are not the discontinuity it tests for.
    Magnitudes are rigid2_dfloor's."""

    variant: ClassVar[str] = "rigid2_dfloor_shape"
    version: ClassVar[str] = "1"
    Params: ClassVar[type[BaseModel]] = StepShapeJumpParams
    params: StepShapeJumpParams

    def __init__(self, params: BaseModel) -> None:
        super().__init__(StepShapeJumpParams.model_validate(params))

    def _null_test(
        self, whitened: NDArray[np.float64], row: int
    ) -> tuple[bool, Mapping[str, object]]:
        rejected, details = super()._null_test(whitened, row)
        if not np.linalg.norm(whitened[row]) >= self.params.shape_level:
            return rejected, details
        fit = fit_shapes(whitened, row)
        gains = {f"shape_gain_{name}": gain for name, gain in fit.gains.items()}
        shape = {"shape": fit.best, "shape_rows": fit.rows} | gains
        return rejected or fit.best != "step", dict(details) | shape


PLUGINS = (StepShapeJumpCheck,)
