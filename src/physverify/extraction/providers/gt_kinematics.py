"""Ground-truth kinematics in LIFT units via the conditional scale fit."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from pydantic import BaseModel

from physverify.extraction._base import GROUND_TRUTH, StageContext, WeightSpec
from physverify.extraction.providers._common import ProviderParams, expect_params
from physverify.extraction.providers._conditional_truth import conditional_truth
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

LENGTH_FIELDS = (
    "position",
    "position_sigma",
    "velocity",
    "velocity_sigma",
    "acceleration",
    "acceleration_sigma",
    "size",
    "size_sigma",
)


class Params(ProviderParams):
    pass


class GroundTruthKinematics:
    stage: ClassVar[Stage] = Stage.KINEMATICS
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

    def run(self, ctx: StageContext) -> ObjectStateTable:
        truth = conditional_truth(ctx)
        scale = truth.scale()
        states = truth.aligned.states
        lengths = {name: getattr(states, name) / scale for name in LENGTH_FIELDS}
        return replace(
            states,
            frame=truth.lift.frame,
            unit=truth.lift.unit,
            source_key=truth.lift.source_key,
            **lengths,
        )


PLUGINS = (GroundTruthKinematics,)
