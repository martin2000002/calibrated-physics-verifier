"""R4_jump rigid2_dfloor_null: rigid2_dfloor that abstains when the track's own stencils away
from the peak reject the white-noise null the statistic is read under."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.checks.variants._declared_depth_jump import DeclaredDepthJumpCheck
from physverify.checks.variants._rigid_jump import RigidJumpParams
from physverify.physics.measures.null_tail import off_peak, tail_test

NULL_LEVEL = 4.0
"""chi_3 level of a stencil counted as a tail event: P(chi_3 > 4) is about 1.1e-3."""
NULL_ALPHA = 0.01
"""Binomial p-value under which the off-peak tail rejects the white-noise null."""
NULL_GUARD = 2
"""Rows each side of the peak left out of the tail test (see ``off_peak``)."""


class NullTailJumpParams(RigidJumpParams):
    null_level: float = Field(default=NULL_LEVEL, gt=0.0)
    null_alpha: float = Field(default=NULL_ALPHA, gt=0.0, lt=1.0)
    null_guard: int = Field(default=NULL_GUARD, ge=0)


class NullTailJumpCheck(DeclaredDepthJumpCheck):
    """R4_jump rigid2_dfloor_null [jump_over_noise]: rigid2_dfloor, but ABSTAIN
    (UNKNOWN_UNCERTAINTY) when the stencils more than ``null_guard`` rows from the peak exceed
    ``null_level`` more often than a binomial test at ``null_alpha`` allows under the chi_3 null
    (``tail_test``).

    sigma-hat is a robust scale, the width of the core of the track's third differences; the
    threshold reads the peak as a chi_3 draw, which needs the rest of the track to have the tails
    of white noise. Sporadic jumps along a track (repeated or irregularly timed frames of a real
    camera, masks that swap region for a frame) break that, and the peak of such a track measures
    them, not a discontinuity of the body. A single jump or excursion stays inside the guard, so
    it does not reject the null it is tested against. Magnitudes are rigid2_dfloor's."""

    variant: ClassVar[str] = "rigid2_dfloor_null"
    version: ClassVar[str] = "1"
    Params: ClassVar[type[BaseModel]] = NullTailJumpParams
    params: NullTailJumpParams

    def __init__(self, params: BaseModel) -> None:
        super().__init__(NullTailJumpParams.model_validate(params))

    def _null_test(
        self, whitened: NDArray[np.float64], row: int
    ) -> tuple[bool, Mapping[str, object]]:
        per_row = np.linalg.norm(whitened, axis=1)
        found = tail_test(off_peak(per_row, row, self.params.null_guard), self.params.null_level)
        details = {
            "null_exceed": found.exceed,
            "null_stencils": found.stencils,
            "null_p_value": found.p_value,
        }
        return found.p_value < self.params.null_alpha, details
