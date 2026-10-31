"""Velocity and acceleration by Savitzky-Golay differentiation with linear sigma. The window is a
duration: at each clip's row rate it becomes the nearest odd row count above the polynomial
order."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from physverify.extraction._base import ProviderParams, StageContext, WeightSpec
from physverify.extraction.providers._common import expect_params
from physverify.physics.kinematics import SavgolSpec, differentiate
from physverify.physics.row_time import odd_rows_in, reference_seconds, seconds_per_row
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable


class Params(ProviderParams):
    window_s: float = Field(default=reference_seconds(7), gt=0.0)
    polyorder: int = Field(default=2, ge=0)

    def spec(self, row_seconds: float) -> SavgolSpec:
        """The window at ``row_seconds`` per row: odd and > polyorder at every frame rate."""
        window = odd_rows_in(self.window_s, row_seconds, self.polyorder + 1)
        return SavgolSpec(window, self.polyorder)


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
        row_seconds = seconds_per_row(lift.frame_index, ctx.sampling.seconds_per_frame)
        return track_derivatives(lift, self.params.spec(row_seconds))


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
