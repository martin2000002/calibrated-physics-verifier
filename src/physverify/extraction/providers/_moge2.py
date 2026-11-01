"""MoGe-2 ViT-L (Wang et al., 2025) runtime shared by the plugins that run it: pinned weights,
loading and batched per-frame inference.

The model code (MIT) and its geometry helpers ``utils3d_moge`` (MIT) are pinned checkouts
imported from ``paths.vendor_dir``; the checkpoint is a pinned Hugging Face snapshot.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from physverify.core.errors import PluginError
from physverify.extraction._base import StageContext, WeightKind, WeightSpec
from physverify.extraction.weights import local_path, vendored_import_path, weight_file

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
WEIGHTS = (CODE, GEOMETRY, CHECKPOINT)
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

    def __init__(self, model: Any, device: str, resolution_level: int, use_fp16: bool) -> None:
        self._model = model
        self._device = device
        self._resolution_level = resolution_level
        self._use_fp16 = use_fp16

    def estimate(self, frames: NDArray[np.uint8], fov_x_deg: float | None) -> FrameEstimate:
        import torch

        image = torch.from_numpy(np.ascontiguousarray(frames)).to(self._device)
        image = image.permute(0, 3, 1, 2).float() / 255.0
        with torch.inference_mode():
            out = self._model.infer(
                image,
                resolution_level=self._resolution_level,
                fov_x=fov_x_deg,
                use_fp16=self._use_fp16,
            )
        return FrameEstimate(
            depth=out["depth"].cpu().numpy().astype(np.float32),
            focal_rel=out["intrinsics"][:, 0, 0].cpu().numpy().astype(np.float64),
        )


def load_model(ctx: StageContext, resolution_level: int, provider_key: str) -> TorchMoGe:
    """``provider_key`` (``stage/name``) labels a checkpoint mismatch."""
    with (
        vendored_import_path(local_path(GEOMETRY, ctx.paths)),
        vendored_import_path(local_path(CODE, ctx.paths)),
    ):
        module = importlib.import_module(VENDORED_MODULE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = module.MoGeModel.from_pretrained(str(weight_file(CHECKPOINT, ctx.paths)))
    check_load_warnings([str(w.message) for w in caught], provider_key)
    use_fp16 = ctx.device != "cpu"
    return TorchMoGe(model.to(ctx.device).eval(), ctx.device, resolution_level, use_fp16)


def check_load_warnings(messages: list[str], provider_key: str) -> None:
    """``from_pretrained`` loads with strict=False and only warns about missing or extra keys."""
    bad = [m for m in messages if any(marker in m for marker in LOAD_WARNINGS)]
    if bad:
        raise PluginError(f"{provider_key}: checkpoint does not match the model: {bad[0][:300]}")
