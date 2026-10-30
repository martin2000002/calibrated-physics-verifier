"""Camera from the depth stage: the depth model's focal, and GeoCalib's gravity (with its sigma)
estimated with that focal as its prior, so focal and gravity agree.

The focal is required: a depth provider that gives none is an error, never a silent fallback to
another focal. ``gravity="none"`` keeps the focal alone (no pose: LIFT then assumes a level
camera, as with camera/assumed_fov).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from physverify.core.errors import MissingInputError
from physverify.extraction._base import ProviderParams, StageContext, WeightSpec
from physverify.extraction.gravity_pose import pose_from_gravity
from physverify.extraction.providers._common import (
    LazyModel,
    expect_params,
    frame_indices,
    set_torch_threads,
)
from physverify.extraction.providers._geocalib import (
    CHECKPOINT,
    CODE,
    Calibration,
    Calibrator,
    SigmaFloor,
    camera_model,
    clip_gravity,
    clip_gravity_sigma_deg,
    load_calibrator,
)
from physverify.schema.perception import CameraModel, DepthMap
from physverify.schema.stages import Stage

GravitySource = Literal["geocalib", "none"]


class Params(ProviderParams):
    gravity: GravitySource = "geocalib"
    n_frames: int = Field(default=8, ge=1)
    focal_sigma_floor_rel: SigmaFloor = math.nan


class DepthFocalCamera:
    stage: ClassVar[Stage] = Stage.CAMERA
    name: ClassVar[str] = "from_depth"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.DEPTH,)
    heavy: ClassVar[bool] = True
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = (CODE, CHECKPOINT)
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)
        self._model: LazyModel[Calibrator] = LazyModel(self.loader())

    def loader(self) -> Callable[[StageContext], Calibrator]:
        return load_calibrator

    def run(self, ctx: StageContext) -> CameraModel:
        depth = ctx.require(Stage.DEPTH, DepthMap)
        focal = depth_focal(ctx, depth)
        floor = self.params.focal_sigma_floor_rel
        camera = camera_model(ctx.sampling, focal, depth.focal_sigma_px, None, floor)
        if self.params.gravity == "none":
            return camera
        result = self._calibrate(ctx, focal)
        return replace(
            camera,
            world_from_camera=pose_from_gravity(clip_gravity(result)),
            gravity_sigma_deg=clip_gravity_sigma_deg(result),
        )

    def _calibrate(self, ctx: StageContext, focal_px: float) -> Calibration:
        s = ctx.sampling
        frames = ctx.video.frames(s.start, s.stop, s.step, s.max_side)
        picked = frames[frame_indices(frames.shape[0], self.params.n_frames)]
        set_torch_threads(ctx.threads)
        return self._model.get(ctx).calibrate(picked, focal_px)


def depth_focal(ctx: StageContext, depth: DepthMap) -> float:
    """The depth stage's focal in this stage's processing pixels."""
    provider = f"depth/{ctx.plan.get(Stage.DEPTH, '?')}"
    if depth.focal_px is None:
        raise MissingInputError(f"{ctx.clip.clip_id}: {provider} gives no focal for camera")
    if depth.sampling.width != ctx.sampling.width:
        raise MissingInputError(
            f"{ctx.clip.clip_id}: {provider} ran at width {depth.sampling.width}, "
            f"camera at {ctx.sampling.width}"
        )
    return depth.focal_px


PLUGINS = (DepthFocalCamera,)
