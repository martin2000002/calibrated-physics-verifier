"""Pinhole camera from an assumed horizontal field of view."""

from __future__ import annotations

import math
from typing import ClassVar

from pydantic import BaseModel, Field

from physverify.extraction._base import StageContext, WeightSpec
from physverify.extraction.providers._common import (
    ProviderParams,
    centred_intrinsics,
    expect_params,
)
from physverify.schema.perception import CameraModel, IntrinsicsSource
from physverify.schema.stages import Stage


class Params(ProviderParams):
    fov_deg: float = Field(default=60.0, gt=0.0, lt=180.0)
    fov_sigma_deg: float = Field(default=10.0, ge=0.0)


class AssumedFovCamera:
    stage: ClassVar[Stage] = Stage.CAMERA
    name: ClassVar[str] = "assumed_fov"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = ()
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> CameraModel:
        sampling = ctx.sampling
        focal = (sampling.width / 2.0) / math.tan(math.radians(self.params.fov_deg) / 2.0)
        return CameraModel(
            sampling=sampling,
            K=centred_intrinsics(focal, sampling),
            source=IntrinsicsSource.ASSUMED_FOV,
            fov_deg=self.params.fov_deg,
            fov_sigma_deg=self.params.fov_sigma_deg,
            world_from_camera=None,
        )
