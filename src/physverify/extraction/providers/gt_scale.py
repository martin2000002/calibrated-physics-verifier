"""Conditional ground-truth scale: least-squares fit of LIFT to the truth."""

from __future__ import annotations

import math
from typing import ClassVar

from pydantic import BaseModel

from physverify.extraction._base import GROUND_TRUTH, StageContext, WeightSpec
from physverify.extraction.providers._common import ProviderParams, expect_params
from physverify.extraction.providers._conditional_truth import conditional_truth
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import Stage


class Params(ProviderParams):
    pass


class GroundTruthScale:
    stage: ClassVar[Stage] = Stage.SCALE
    name: ClassVar[str] = GROUND_TRUTH
    version: ClassVar[str] = "2"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.TRACK, Stage.LIFT)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = True
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ScaleEstimate:
        truth = conditional_truth(ctx)
        if truth.fit is None:
            return _unknown(truth.lift.source_key, "no_pairs")
        if math.isnan(truth.fit.sigma):
            # A ground_truth scale needs a finite sigma (schema/scale.py); one ratio has no spread.
            return _unknown(truth.lift.source_key, "single_ratio")
        return ScaleEstimate(
            meters_per_unit=truth.fit.meters_per_unit,
            sigma=truth.fit.sigma,
            source=ScaleSource.GROUND_TRUTH,
            lift_key=truth.lift.source_key,
            detail="conditional_fit",
        )


def _unknown(lift_key: str, detail: str) -> ScaleEstimate:
    return ScaleEstimate(
        meters_per_unit=None,
        sigma=math.nan,
        source=ScaleSource.NONE,
        lift_key=lift_key,
        detail=detail,
    )


PLUGINS = (GroundTruthScale,)
