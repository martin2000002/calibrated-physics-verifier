"""Ground-truth detections of the first sampled frame."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from physverify.extraction._base import GROUND_TRUTH, ProviderParams, StageContext, WeightSpec
from physverify.extraction.providers._common import expect_params
from physverify.schema.perception import Detections
from physverify.schema.stages import Stage


class Params(ProviderParams):
    pass


class GroundTruthDetect:
    stage: ClassVar[Stage] = Stage.DETECT
    name: ClassVar[str] = GROUND_TRUTH
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = ()
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = True
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> Detections:
        return ctx.require_gt().detections(ctx.sampling)


PLUGINS = (GroundTruthDetect,)
