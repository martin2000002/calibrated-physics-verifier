"""Back-projection whose sphere tracks carry an along-ray sigma from their apparent-size
constancy (``extraction/size_constancy.py``); positions and sizes are the parent lift's."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import numpy as np
from pydantic import BaseModel

from physverify.extraction._base import StageContext
from physverify.extraction.lift_params import SizeConstancyLiftParams, size_constancy_params
from physverify.extraction.providers._backproject import BackprojectLift, CentreRegistry
from physverify.extraction.providers._common import expect_params
from physverify.extraction.size_constancy import along_ray_sigma, replace_along_ray_sigma
from physverify.schema.states import ObjectStateTable, PrimitiveKind


class SizeConstancyLift(BackprojectLift):
    """``lift/backproject`` with ``position_sigma`` rewritten along the ray on the clean rows of
    sphere tracks; every other field and cell is left as the parent computed it."""

    name: ClassVar[str] = "backproject_size_constancy"
    # 2: the inherited output gains nearer_cover.
    # 3: the inherited image-plane centre comes from a configured strategy.
    # 4: the inherited output gains causal_centre.
    version: ClassVar[str] = "4"
    Params: ClassVar[type[BaseModel]] = SizeConstancyLiftParams

    def __init__(self, params: BaseModel, centres: CentreRegistry) -> None:
        own = expect_params(params, SizeConstancyLiftParams)
        super().__init__(own, centres)
        self.constancy = size_constancy_params(own)

    def run(self, ctx: StageContext) -> ObjectStateTable:
        table = super().run(ctx)
        spheres = np.asarray(table.primitive) == PrimitiveKind.SPHERE.value
        along = along_ray_sigma(
            np.asarray(self._track(ctx).masks, dtype=np.bool_),
            np.asarray(table.position[..., 2], dtype=np.float64),
            np.asarray(table.visible, dtype=np.bool_),
            spheres,
            self.constancy,
        )
        sigma = replace_along_ray_sigma(table.position, table.position_sigma, along)
        return replace(table, position_sigma=sigma)


PLUGINS = (SizeConstancyLift,)
