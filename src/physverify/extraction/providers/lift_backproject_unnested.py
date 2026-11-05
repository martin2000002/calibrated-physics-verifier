"""Back-projection of the tracks that are not nested in another track, with the real-video
label primitives."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from physverify.extraction._base import StageContext
from physverify.extraction.lift_params import UnnestedLiftParams
from physverify.extraction.providers._backproject import BackprojectLift, CentreRegistry
from physverify.extraction.providers._common import expect_params
from physverify.extraction.providers._nesting import without_nested
from physverify.schema.perception import MaskTrack


class UnnestedBackprojectLift(BackprojectLift):
    """``lift/backproject`` on the tracks ``without_nested`` keeps (``nested_overlap``), with
    ``REAL_PRIMITIVES`` as the default label map."""

    name: ClassVar[str] = "backproject_unnested"
    # 3: 1c8de44 changed the inherited footprint yaw trim; cached v2 lifts went stale.
    # 4: the inherited output gains nearer_cover.
    # 5: the inherited image-plane centre comes from a configured strategy.
    version: ClassVar[str] = "5"
    Params: ClassVar[type[BaseModel]] = UnnestedLiftParams

    def __init__(self, params: BaseModel, centres: CentreRegistry) -> None:
        unnested = expect_params(params, UnnestedLiftParams)
        super().__init__(unnested, centres)
        self.nested_overlap = unnested.nested_overlap

    def _track(self, ctx: StageContext) -> MaskTrack:
        return without_nested(super()._track(ctx), self.nested_overlap)


PLUGINS = (UnnestedBackprojectLift,)
