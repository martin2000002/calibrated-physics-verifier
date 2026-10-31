"""Ground-truth depth: metric z, or relative with object median 1 (default); both carry the
camera's true focal, so a camera taken from the depth stage works on true depth.

Standing in for a perceived depth provider, ``metric`` follows that provider's declared
``depth_output``: metric z for a metric model, so the chain downstream (a metric lift, its
metric-depth scale) behaves as with the model and only the depth error is removed."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar, Final

import numpy as np
from pydantic import BaseModel

from physverify.extraction._base import (
    GROUND_TRUTH,
    ProviderParams,
    StageContext,
    StageProvider,
    WeightSpec,
)
from physverify.extraction.backprojection import normalize_by_object_median
from physverify.extraction.plan import PerceivedParams
from physverify.extraction.providers._common import expect_params
from physverify.schema.perception import DepthKind, DepthMap
from physverify.schema.stages import Stage

DEPTH_OUTPUT_ATTR: Final = "depth_output"
"""ClassVar ``DepthKind`` a depth provider declares for the maps it returns."""


class Params(ProviderParams):
    metric: bool = False


def outputs_metric_depth(perceived: type[StageProvider]) -> bool:
    return getattr(perceived, DEPTH_OUTPUT_ATTR, None) is DepthKind.METRIC_Z


class GroundTruthDepth:
    stage: ClassVar[Stage] = Stage.DEPTH
    name: ClassVar[str] = GROUND_TRUTH
    version: ClassVar[str] = "2"
    requires: ClassVar[tuple[Stage, ...]] = ()
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = True
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params
    perceived_params: ClassVar[PerceivedParams] = {"metric": outputs_metric_depth}

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
        return replace(
            depth,
            depth=np.where(depth.valid, relative, 0.0).astype(np.float32),
            kind=DepthKind.RELATIVE_DEPTH,
        )


PLUGINS = (GroundTruthDepth,)
