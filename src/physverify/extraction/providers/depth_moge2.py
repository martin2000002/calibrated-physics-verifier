"""MoGe-2 ViT-L (Wang et al., 2025): metric depth and the horizontal focal from single frames.

The runtime (pinned weights, loading, inference) lives in ``_moge2``. The depth is a learned
metric estimate, stored as metric z: its error against the truth is what the depth and scale
diagnoses measure, and the model gives no uncertainty for its focal (sigma NaN).

Frames are independent, so a first pass on ``shared_focal_frames`` evenly spaced frames fixes
one focal (their median) and the second pass predicts every frame with that field of view.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import replace
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.extraction._base import ProviderParams, StageContext, WeightSpec
from physverify.extraction.providers._common import (
    LazyModel,
    expect_params,
    frame_indices,
    set_torch_threads,
)
from physverify.extraction.providers._moge2 import (
    MAX_RESOLUTION_LEVEL,
    WEIGHTS,
    PointMapModel,
    load_model,
    predict,
)
from physverify.extraction.providers._vda import to_depth_map
from physverify.schema.perception import DepthKind, DepthMap, horizontal_fov_deg
from physverify.schema.stages import Stage

OWNER = "depth/moge2"


class Params(ProviderParams):
    resolution_level: int = Field(default=MAX_RESOLUTION_LEVEL, ge=0, le=MAX_RESOLUTION_LEVEL)
    shared_focal_frames: int = Field(default=8, ge=0)
    batch_size: int = Field(default=4, ge=1)


class MoGe2Depth:
    stage: ClassVar[Stage] = Stage.DEPTH
    name: ClassVar[str] = "moge2"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = ()
    heavy: ClassVar[bool] = True
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = WEIGHTS
    Params: ClassVar[type[BaseModel]] = Params
    depth_output: ClassVar[DepthKind] = DepthKind.METRIC_Z

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)
        self._model: LazyModel[PointMapModel] = LazyModel(self.loader())

    def loader(self) -> Callable[[StageContext], PointMapModel]:
        return functools.partial(
            load_model, resolution_level=self.params.resolution_level, owner=OWNER
        )

    def run(self, ctx: StageContext) -> DepthMap:
        s = ctx.sampling
        frames = ctx.video.frames(s.start, s.stop, s.step, s.max_side)
        set_torch_threads(ctx.threads)
        model = self._model.get(ctx)
        fov_x = self._shared_fov(model, frames)
        estimate = predict(model, frames, fov_x, self.params.batch_size)
        focal_rel = float(np.median(estimate.focal_rel))
        depth = to_depth_map(s, estimate.depth, self.depth_output)
        return replace(depth, focal_px=focal_rel * s.width, focal_sigma_px=math.nan)

    def _shared_fov(self, model: PointMapModel, frames: NDArray[np.uint8]) -> float | None:
        if self.params.shared_focal_frames == 0:
            return None
        picked = frames[frame_indices(frames.shape[0], self.params.shared_focal_frames)]
        probe = predict(model, picked, None, self.params.batch_size)
        return horizontal_fov_deg(float(np.median(probe.focal_rel)), 1.0)


PLUGINS = (MoGe2Depth,)
