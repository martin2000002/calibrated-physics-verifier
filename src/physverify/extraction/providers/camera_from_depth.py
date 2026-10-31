"""Camera from the depth stage: the depth model's focal, and GeoCalib's gravity (with its sigma)
estimated with that focal as its prior, so focal and gravity agree.

The focal is required: a depth provider that gives none is an error, never a silent fallback to
another focal. ``gravity="none"`` keeps the focal alone (no pose: LIFT then assumes a level
camera, as with camera/assumed_fov).

The focal's relative sigma phi is the depth model's measured ``focal_rel`` (rms of bias and sigma,
the DEPTH entry of the error profile), at least ``focal_sigma_floor_rel``. Without that entry it is
``unmeasured_focal_sigma_rel``, by default the phi of camera/assumed_fov's field of view: an
unmeasured focal counts as no better than an assumed one. It is never NaN, so LIFT always gets a
known focal sigma from this camera. On the true depth the focal is the true one: phi is 0 and the
floor, which bounds a model's error, does not apply.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel, Field

from physverify.core.errors import MissingInputError
from physverify.extraction._base import GROUND_TRUTH, ProviderParams, StageContext, WeightSpec
from physverify.extraction.providers._common import (
    LazyModel,
    expect_params,
    frame_indices,
    set_torch_threads,
)
from physverify.extraction.providers._geocalib import (
    CHECKPOINT,
    CODE,
    UNMEASURED_FOCAL_SIGMA_REL,
    Calibration,
    Calibrator,
    GravitySource,
    camera_model,
    load_calibrator,
    with_gravity,
)
from physverify.physics.uncertainty import entry_rms
from physverify.schema.error_profile import FOCAL_REL
from physverify.schema.perception import CameraModel, DepthMap
from physverify.schema.stages import Stage


class Params(ProviderParams):
    gravity: GravitySource = "geocalib"
    n_frames: int = Field(default=8, ge=1)
    focal_sigma_floor_rel: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    unmeasured_focal_sigma_rel: float = Field(
        default=UNMEASURED_FOCAL_SIGMA_REL, gt=0.0, allow_inf_nan=False
    )


class DepthFocalCamera:
    stage: ClassVar[Stage] = Stage.CAMERA
    name: ClassVar[str] = "from_depth"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.DEPTH,)
    heavy: ClassVar[bool] = True
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = (Stage.DEPTH,)
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
        camera = camera_model(ctx.sampling, focal, self._focal_phi(ctx), None)
        if self.params.gravity == "none":
            return camera
        return with_gravity(camera, self._calibrate(ctx, focal))

    def _focal_phi(self, ctx: StageContext) -> float:
        if ctx.plan.get(Stage.DEPTH) == GROUND_TRUTH:
            return 0.0
        measured = entry_rms(ctx.error_entry(Stage.DEPTH, FOCAL_REL))
        phi = self.params.unmeasured_focal_sigma_rel if math.isnan(measured) else measured
        return max(phi, self.params.focal_sigma_floor_rel)

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
