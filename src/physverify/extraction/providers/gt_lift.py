"""Ground-truth states in camera frame, sampled frames and TRACK ids, in meters."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from pydantic import BaseModel

from physverify.extraction._base import GROUND_TRUTH, StageContext, WeightSpec
from physverify.extraction.matching import align_ground_truth
from physverify.extraction.pipeline import PENDING_SOURCE_KEY
from physverify.extraction.providers._common import ProviderParams, expect_params
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import Stage
from physverify.schema.states import LengthUnit, ObjectStateTable


class Params(ProviderParams):
    pass


class GroundTruthLift:
    stage: ClassVar[Stage] = Stage.LIFT
    name: ClassVar[str] = GROUND_TRUTH
    version: ClassVar[str] = "2"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.TRACK,)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = True
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ObjectStateTable:
        track = ctx.require(Stage.TRACK, MaskTrack)
        aligned = align_ground_truth(ctx.require_gt(), track, ctx.sampling)
        return replace(aligned.states, unit=LengthUnit.METER, source_key=PENDING_SOURCE_KEY)


PLUGINS = (GroundTruthLift,)
