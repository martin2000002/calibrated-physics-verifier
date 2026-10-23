"""EdgeTAM box-prompted video masks, the fast-profile tracker.

EdgeTAM's config builds its RepViT backbone config from ``timm/repvit_m1.dist_in1k`` through the
hub cache, so that config is pinned here too and loaded offline via its ``refs/main``.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from physverify.extraction._base import StageContext, WeightKind, WeightSpec
from physverify.extraction.providers._common import expect_params, set_torch_threads
from physverify.extraction.providers._sam_video import (
    VideoModel,
    VideoTrackParams,
    run_video_tracker,
    video_model_loader,
)
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import Stage

MODEL = VideoModel(
    "EdgeTamVideoModel",
    WeightSpec(
        WeightKind.HF,
        "yonigozlan/EdgeTAM-hf",
        "c266ce53b3fc00f0f495b583f6a116c4e57f53bb",
        "8858f8e4757b0b96dab8763f296ecffd845efbbbf698f64163cfa20a63d5fff4",
    ),
)
BACKBONE_CONFIG = WeightSpec(
    WeightKind.HF,
    "timm/repvit_m1.dist_in1k/config.json",
    "55e64695f26bb1711730f916e557b464fb46478d",
    "dd51f1dda78893418f737a47fb2755e2a4a76783ea31aac7ce195e6dd54d7fe9",
)


class Params(VideoTrackParams):
    pass


class EdgeTamTracker:
    stage: ClassVar[Stage] = Stage.TRACK
    name: ClassVar[str] = "edgetam"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.DETECT,)
    heavy: ClassVar[bool] = True
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = (MODEL.weights, BACKBONE_CONFIG)
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)
        self._model = video_model_loader(MODEL)

    def run(self, ctx: StageContext) -> MaskTrack:
        set_torch_threads(ctx.threads)
        return run_video_tracker(ctx, self._model, self.params.mask_threshold)


PLUGINS = (EdgeTamTracker,)
