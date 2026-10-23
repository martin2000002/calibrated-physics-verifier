"""Velocity and acceleration by Savitzky-Golay differentiation with linear sigma."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field, model_validator

from physverify.extraction._base import StageContext, WeightSpec
from physverify.extraction.providers._common import ProviderParams, expect_params
from physverify.physics.kinematics import SavgolSpec, differentiate
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable


class Params(ProviderParams):
    window: int = Field(default=7, ge=1)
    polyorder: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def _check_spec(self) -> Params:
        SavgolSpec(self.window, self.polyorder)
        return self


class SavgolKinematics:
    stage: ClassVar[Stage] = Stage.KINEMATICS
    name: ClassVar[str] = "savgol"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.LIFT,)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ObjectStateTable:
        lift = ctx.require(Stage.LIFT, ObjectStateTable)
        return track_derivatives(lift, SavgolSpec(self.params.window, self.params.polyorder))


def track_derivatives(lift: ObjectStateTable, spec: SavgolSpec) -> ObjectStateTable:
    """LIFT table with v, a and their sigma filled object by object; positions untouched."""
    names = ("velocity", "acceleration", "velocity_sigma", "acceleration_sigma")
    out: dict[str, Any] = {name: np.full(lift.position.shape, np.nan) for name in names}
    for n in range(lift.n_objects):
        valid = lift.visible[:, n] & np.all(np.isfinite(lift.position[:, n]), axis=1)
        derivatives = differentiate(
            lift.t, lift.position[:, n], lift.position_sigma[:, n], valid, spec
        )
        for name in names:
            out[name][:, n] = getattr(derivatives, name)
    return replace(lift, **out)


PLUGINS = (SavgolKinematics,)
