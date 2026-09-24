"""Conditional ground truth expressed in the perceived ids and LIFT units (ADR 0005, 3.8)."""

from __future__ import annotations

from dataclasses import dataclass

from physverify.core.errors import MissingInputError
from physverify.extraction._base import StageContext
from physverify.extraction.matching import (
    AlignedTruth,
    ScaleFit,
    align_ground_truth,
    conditional_scale_fit,
)
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable


@dataclass(frozen=True)
class ConditionalTruth:
    lift: ObjectStateTable
    aligned: AlignedTruth
    fit: ScaleFit | None

    def scale(self) -> float:
        """Meters per LIFT unit from the conditional fit (never from SCALE)."""
        if self.fit is None:
            raise MissingInputError(
                f"{self.lift.clip_id}: no matched visible object to fit the conditional scale"
            )
        return self.fit.meters_per_unit


def conditional_truth(ctx: StageContext) -> ConditionalTruth:
    track = ctx.require(Stage.TRACK, MaskTrack)
    lift = ctx.require(Stage.LIFT, ObjectStateTable)
    aligned = align_ground_truth(ctx.require_gt(), track, ctx.sampling)
    return ConditionalTruth(
        lift=lift, aligned=aligned, fit=conditional_scale_fit(lift, aligned.states)
    )
