"""SAM 2.1 hiera-tiny box-prompted video masks, the precise-profile tracker."""

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
    "Sam2VideoModel",
    WeightSpec(
        WeightKind.HF,
        "facebook/sam2.1-hiera-tiny",
        "de431c4043854a71d8101e17995dfe596bf101a5",
        "48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7",
    ),
)


class Params(VideoTrackParams):
    pass


class Sam2Tracker:
    stage: ClassVar[Stage] = Stage.TRACK
    name: ClassVar[str] = "sam2_1_hiera_tiny"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.DETECT,)
    heavy: ClassVar[bool] = True
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = (MODEL.weights,)
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)
        self._model = video_model_loader(MODEL)

    def run(self, ctx: StageContext) -> MaskTrack:
        set_torch_threads(ctx.threads)
        return run_video_tracker(ctx, self._model, self.params.mask_threshold)


PLUGINS = (Sam2Tracker,)
