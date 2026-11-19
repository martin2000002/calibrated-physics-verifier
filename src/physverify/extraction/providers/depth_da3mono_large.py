"""Depth Anything 3 Monocular Large (DA3MONO-LARGE; Lin et al., 2025): relative depth per frame.

The network predicts depth up to an unknown scale, frame by frame; its reciprocal is stored as
affine disparity, a learned estimate whose error the depth diagnoses measure, never metric depth.
Frames larger than ``max_input_side`` are area-downsampled to it first; DA3 then resizes them to
``process_res`` itself. The defaults are the configuration whose calibration was measured.

Neither the paper's training table nor its teacher's synthetic data names Kubric or MOVi (Lin et
al., 2025, table 1 and section 4.1); the unlabelled images of the monocular student are not
listed. Code and weights are Apache-2.0.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel, Field

from physverify.extraction._base import ProviderParams, StageContext, WeightKind, WeightSpec
from physverify.extraction.providers._common import LazyModel, expect_params, set_torch_threads
from physverify.extraction.providers._da3 import (
    CODE,
    DepthModel,
    load_model,
    predict_disparity,
    reduce_frames,
    set_cv2_threads,
)
from physverify.extraction.providers._vda import to_depth_map
from physverify.schema.perception import DepthKind, DepthMap
from physverify.schema.stages import Stage

PROVIDER_KEY = "depth/da3mono_large"
MODEL_NAME = "da3mono-large"
CHECKPOINT = WeightSpec(
    WeightKind.HF,
    "depth-anything/DA3MONO-LARGE/model.safetensors",
    "f465978e618db8cc79c83b8bbf24964857db1875",
    "7a799a7f95eb8d4c404c2ca8be3dc3276b350a417ddc4420db72ba850cc0e960",
)


class Params(ProviderParams):
    process_res: int = Field(default=504, ge=14, multiple_of=14)
    max_input_side: int = Field(default=512, ge=14)


class DA3MonoLarge:
    stage: ClassVar[Stage] = Stage.DEPTH
    name: ClassVar[str] = "da3mono_large"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = ()
    heavy: ClassVar[bool] = True
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = (CODE, CHECKPOINT)
    Params: ClassVar[type[BaseModel]] = Params
    depth_output: ClassVar[DepthKind] = DepthKind.AFFINE_DISPARITY

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)
        self._model: LazyModel[DepthModel] = LazyModel(self.loader())

    def loader(self) -> Callable[[StageContext], DepthModel]:
        return functools.partial(
            load_model,
            checkpoint=CHECKPOINT,
            model_name=MODEL_NAME,
            process_res=self.params.process_res,
            provider_key=PROVIDER_KEY,
        )

    def run(self, ctx: StageContext) -> DepthMap:
        s = ctx.sampling
        frames = ctx.video.frames(s.start, s.stop, s.step, s.max_side)
        set_torch_threads(ctx.threads)
        set_cv2_threads(ctx.threads)
        model = self._model.get(ctx)
        disparity = predict_disparity(model, reduce_frames(frames, self.params.max_input_side))
        return to_depth_map(s, disparity, self.depth_output)


PLUGINS = (DA3MonoLarge,)
