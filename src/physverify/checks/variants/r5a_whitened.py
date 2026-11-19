"""R5a whitened: v1's candidates, each scored by the Wald contrast of the two accelerations fitted
with weights from the per-row position sigma instead of their spread over |a|."""

from __future__ import annotations

from physverify.checks.variants._whitened import (
    EXACT_SIGMA_LENGTHS,
    WhitenedFallShapeCheck,
    WhitenedFallShapeParams,
)

__all__ = ("EXACT_SIGMA_LENGTHS", "WhitenedFallShapeCheck", "WhitenedFallShapeParams")

PLUGINS = (WhitenedFallShapeCheck,)
