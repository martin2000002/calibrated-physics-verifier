"""Back-projection whose depth along each track follows the track's apparent mask area
(``extraction/area_depth.py``); every other field is the parent lift's."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from physverify.extraction._base import StageContext
from physverify.extraction.area_depth import (
    area_depth_factor,
    area_depth_rows,
    mask_area,
    scale_depth_fields,
)
from physverify.extraction.lift_params import LiftParams
from physverify.extraction.providers._backproject import BackprojectLift
from physverify.extraction.providers._common import expect_params
from physverify.schema.states import ObjectStateTable


class AreaDepthLiftParams(LiftParams):
    """Fixed before any measurement and kept out of the profile tables, so no section or
    run-config hash moves; it lives here so the shared lift settings, and with them the source
    pins of the other lifts, stay as they are."""

    # The smallest count whose median is one of the rows rather than the mean of two.
    min_rows: int = Field(default=3, ge=1)


class AreaDepthLift(BackprojectLift):
    """``lift/backproject`` with position, size and their sigmas rescaled along the camera ray on
    the trusted rows of each track, perceived or true.

    Its params come from ``[extraction.providers.lift.backproject_area_depth]`` alone: as an
    input route's lift it does not inherit overrides of the main lift's table."""

    name: ClassVar[str] = "backproject_area_depth"
    # 2: the inherited image-plane centre comes from a configured strategy.
    version: ClassVar[str] = "2"
    Params: ClassVar[type[BaseModel]] = AreaDepthLiftParams
    derives_depth_from_size: ClassVar[bool] = True

    def __init__(self, params: BaseModel) -> None:
        own = expect_params(params, AreaDepthLiftParams)
        super().__init__(own)
        self.min_rows = own.min_rows

    def run(self, ctx: StageContext) -> ObjectStateTable:
        table = super().run(ctx)
        area = mask_area(np.asarray(self._track(ctx).masks, dtype=np.bool_))
        used = area_depth_rows(table, area)
        position_z = np.asarray(table.position[..., 2], dtype=np.float64)
        return scale_depth_fields(table, area_depth_factor(position_z, area, used, self.min_rows))


PLUGINS = (AreaDepthLift,)
