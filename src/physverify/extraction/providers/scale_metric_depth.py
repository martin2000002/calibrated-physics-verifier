"""Scale straight from a metric depth: the lift is already in metres, so one metre per unit.

The depth is a learned estimate, not a measurement: the sigma is this provider's measured
``scale_rel`` error (rms of bias and sigma) from the error profile, NaN until it is measured.
It reads no size prior, so the generator's size distribution cannot leak into the scale.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from physverify.core.errors import MissingInputError
from physverify.extraction._base import ProviderParams, StageContext, WeightSpec
from physverify.extraction.providers._common import expect_params
from physverify.physics.uncertainty import entry_rms
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import Stage
from physverify.schema.states import LengthUnit, ObjectStateTable

METERS_PER_METER = 1.0


class Params(ProviderParams):
    pass


class MetricDepthScale:
    stage: ClassVar[Stage] = Stage.SCALE
    name: ClassVar[str] = "metric_depth"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.LIFT,)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = (Stage.SCALE,)
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ScaleEstimate:
        lift = ctx.require(Stage.LIFT, ObjectStateTable)
        if lift.unit is not LengthUnit.METER:
            raise MissingInputError(
                f"{ctx.clip.clip_id}: scale/metric_depth needs a metric lift, got "
                f"{lift.unit.value} (depth/{ctx.plan.get(Stage.DEPTH, '?')} is not metric)"
            )
        rel_sigma = entry_rms(ctx.error_entry(Stage.SCALE, "scale_rel"))
        return ScaleEstimate(
            meters_per_unit=METERS_PER_METER,
            sigma=rel_sigma * METERS_PER_METER,
            source=ScaleSource.METRIC_DEPTH,
            lift_key=lift.source_key,
            detail="metric_depth",
        )


PLUGINS = (MetricDepthScale,)
