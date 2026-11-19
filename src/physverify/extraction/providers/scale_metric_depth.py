"""Scale straight from a metric depth: the lift is already in metres, so one metre per unit.

The depth is a learned estimate, not a measurement. The sigma is the depth model's measured
``scale_rel`` (rms of bias and sigma); NaN until measured, 0 on true depth. Under the focal
convention of ``_size_prior`` a metric depth has k = 0: z does not depend on f, and the
focal error of lateral metres ``(u - cx) z / f`` is LIFT's phi, never counted again here. A
relative lift has no scale: the provider abstains (``source=none``) instead of claiming one. It
reads no size prior, so the generator's size distribution cannot leak into the scale.
"""

from __future__ import annotations

import math
from typing import ClassVar

from pydantic import BaseModel

from physverify.extraction._base import ProviderParams, StageContext, WeightSpec
from physverify.extraction.providers._common import expect_params
from physverify.physics.uncertainty import entry_rms
from physverify.schema.error_profile import SCALE_REL
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import Stage
from physverify.schema.states import LengthUnit, ObjectStateTable

METERS_PER_METER = 1.0


class Params(ProviderParams):
    pass


class MetricDepthScale:
    stage: ClassVar[Stage] = Stage.SCALE
    name: ClassVar[str] = "metric_depth"
    version: ClassVar[str] = "3"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.LIFT,)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = (Stage.DEPTH,)
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ScaleEstimate:
        lift = ctx.require(Stage.LIFT, ObjectStateTable)
        if lift.unit is not LengthUnit.METER:
            return ScaleEstimate(None, math.nan, ScaleSource.NONE, lift.source_key, "relative_lift")
        rel_sigma = entry_rms(ctx.error_entry(Stage.DEPTH, SCALE_REL))
        return ScaleEstimate(
            meters_per_unit=METERS_PER_METER,
            sigma=rel_sigma * METERS_PER_METER,
            source=ScaleSource.METRIC_DEPTH,
            lift_key=lift.source_key,
            detail="metric_depth",
        )


PLUGINS = (MetricDepthScale,)
