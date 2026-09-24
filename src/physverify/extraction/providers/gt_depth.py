"""Ground-truth depth: metric z, or relative with object median 1 (default)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel

from physverify.extraction._base import GROUND_TRUTH, StageContext, WeightSpec
from physverify.extraction.providers._common import ProviderParams, expect_params
from physverify.extraction.providers._lifting import normalize_by_object_median
from physverify.schema.perception import DepthKind, DepthMap
from physverify.schema.stages import Stage


class Params(ProviderParams):
    metric: bool = False


class GroundTruthDepth:
    stage: ClassVar[Stage] = Stage.DEPTH
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

    def run(self, ctx: StageContext) -> DepthMap:
        gt = ctx.require_gt()
        depth = gt.depth(ctx.sampling)
        if self.params.metric:
            return depth
        objects = gt.masks(ctx.sampling).masks.any(axis=1)
        relative, _ = normalize_by_object_median(
            depth.depth.astype(np.float64), depth.valid, objects
        )
        return DepthMap(
            sampling=ctx.sampling,
            depth=np.where(depth.valid, relative, 0.0).astype(np.float32),
            kind=DepthKind.RELATIVE_DEPTH,
            valid=depth.valid,
        )


PLUGINS = (GroundTruthDepth,)
