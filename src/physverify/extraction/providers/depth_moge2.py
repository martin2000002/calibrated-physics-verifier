"""MoGe-2 ViT-L (Wang et al., 2025): metric depth and the horizontal focal from single frames.

The model code (MIT) and its geometry helpers ``utils3d_moge`` (MIT) are pinned checkouts
imported from ``paths.vendor_dir``; the checkpoint is a pinned Hugging Face snapshot. The depth
is a learned metric estimate, stored as metric z: its error against the truth is what the depth
and scale diagnoses measure, and the model gives no uncertainty for its focal (sigma NaN).

Frames are independent, so a first pass on ``shared_focal_frames`` evenly spaced frames fixes
one focal (their median) and the second pass predicts every frame with that field of view.
"""

from __future__ import annotations

import functools
import importlib
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from physverify.core.errors import PluginError
from physverify.extraction._base import ProviderParams, StageContext, WeightKind, WeightSpec
from physverify.extraction.providers._common import (
    LazyModel,
    expect_params,
    frame_indices,
    set_torch_threads,
)
from physverify.extraction.providers._vda import to_depth_map
from physverify.extraction.weights import local_path, vendored_import_path, weight_file
from physverify.schema.perception import DepthKind, DepthMap, horizontal_fov_deg
from physverify.schema.stages import Stage

CODE = WeightSpec(
    WeightKind.GIT,
    "https://github.com/microsoft/MoGe.git",
    "74fbce054ebed49800de42d0ad0e83495065719a",
)
GEOMETRY = WeightSpec(
    WeightKind.GIT,
    "https://github.com/EasternJournalist/utils3d-moge.git",
    "62f09d58509485564e24d5d9f6aac9ee9ebc0c37",
)
CHECKPOINT = WeightSpec(
    WeightKind.HF,
    "Ruicheng/moge-2-vitl/model.pt",
    "39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
    "3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5",
)
VENDORED_MODULE = "moge.model.v2"
LOAD_WARNINGS = ("absent from the checkpoint", "have no counterpart in the model")
MAX_RESOLUTION_LEVEL = 9


@dataclass(frozen=True)
class FrameEstimate:
    """Per-frame metric z (non-finite where the model masks the pixel) and focal / width."""

    depth: NDArray[np.float32]
    focal_rel: NDArray[np.float64]


class PointMapModel(Protocol):
    def estimate(self, frames: NDArray[np.uint8], fov_x_deg: float | None) -> FrameEstimate: ...


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
    weights: ClassVar[tuple[WeightSpec, ...]] = (CODE, GEOMETRY, CHECKPOINT)
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)
        self._model: LazyModel[PointMapModel] = LazyModel(self.loader())

    def loader(self) -> Callable[[StageContext], PointMapModel]:
        return functools.partial(load_model, resolution_level=self.params.resolution_level)

    def run(self, ctx: StageContext) -> DepthMap:
        s = ctx.sampling
        frames = ctx.video.frames(s.start, s.stop, s.step, s.max_side)
        set_torch_threads(ctx.threads)
        model = self._model.get(ctx)
        fov_x = self._shared_fov(model, frames)
        estimate = predict(model, frames, fov_x, self.params.batch_size)
        focal_rel = float(np.median(estimate.focal_rel))
        depth = to_depth_map(s, estimate.depth, DepthKind.METRIC_Z)
        return replace(depth, focal_px=focal_rel * s.width, focal_sigma_px=math.nan)

    def _shared_fov(self, model: PointMapModel, frames: NDArray[np.uint8]) -> float | None:
        if self.params.shared_focal_frames == 0:
            return None
        picked = frames[frame_indices(frames.shape[0], self.params.shared_focal_frames)]
        probe = predict(model, picked, None, self.params.batch_size)
        return horizontal_fov_deg(float(np.median(probe.focal_rel)), 1.0)


def predict(
    model: PointMapModel, frames: NDArray[np.uint8], fov_x: float | None, batch_size: int
) -> FrameEstimate:
    parts = [
        model.estimate(frames[start : start + batch_size], fov_x)
        for start in range(0, frames.shape[0], batch_size)
    ]
    return FrameEstimate(
        depth=np.concatenate([p.depth for p in parts]).astype(np.float32),
        focal_rel=np.concatenate([p.focal_rel for p in parts]).astype(np.float64),
    )


class TorchMoGe:
    """The vendored ``MoGeModel`` on uint8 RGB frames; intrinsics are normalised by image size."""

    def __init__(self, model: Any, device: str, resolution_level: int) -> None:
        self._model = model
        self._device = device
        self._resolution_level = resolution_level

    def estimate(self, frames: NDArray[np.uint8], fov_x_deg: float | None) -> FrameEstimate:
        import torch

        image = torch.from_numpy(np.ascontiguousarray(frames)).to(self._device)
        image = image.permute(0, 3, 1, 2).float() / 255.0
        with torch.inference_mode():
            out = self._model.infer(
                image,
                resolution_level=self._resolution_level,
                fov_x=fov_x_deg,
                use_fp16=self._device != "cpu",
            )
        return FrameEstimate(
            depth=out["depth"].cpu().numpy().astype(np.float32),
            focal_rel=out["intrinsics"][:, 0, 0].cpu().numpy().astype(np.float64),
        )


def load_model(ctx: StageContext, resolution_level: int) -> TorchMoGe:
    with (
        vendored_import_path(local_path(GEOMETRY, ctx.paths)),
        vendored_import_path(local_path(CODE, ctx.paths)),
    ):
        module = importlib.import_module(VENDORED_MODULE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = module.MoGeModel.from_pretrained(str(weight_file(CHECKPOINT, ctx.paths)))
    check_load_warnings([str(w.message) for w in caught])
    return TorchMoGe(model.to(ctx.device).eval(), ctx.device, resolution_level)


def check_load_warnings(messages: list[str]) -> None:
    """``from_pretrained`` loads with strict=False and only warns about missing or extra keys."""
    bad = [m for m in messages if any(marker in m for marker in LOAD_WARNINGS)]
    if bad:
        raise PluginError(f"depth/moge2: checkpoint does not match the model: {bad[0][:300]}")


PLUGINS = (MoGe2Depth,)
