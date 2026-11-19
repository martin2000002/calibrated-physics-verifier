"""Depth Anything 3 (Lin et al., 2025) runtime shared by the plugins that run it: pinned code and
weights, loading, and single-frame inference as ``DepthAnything3.inference`` runs it (api.py).

The model code (Apache-2.0) is a pinned checkout whose ``src`` is imported from
``paths.vendor_dir``. Its ``api`` module is never imported: it pulls the export and
pose-alignment packages (moviepy, pycolmap, gsplat, evo), which single-image inference without
poses never calls, so the network, preprocessing and output processing are taken from their own
modules. The checkpoint (Apache-2.0) is a pinned Hugging Face snapshot.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from physverify.core.errors import PluginError
from physverify.core.resample import resize_frame
from physverify.extraction._base import StageContext, WeightKind, WeightSpec
from physverify.extraction.providers._vda import check_vendored_package
from physverify.extraction.weights import local_path, vendored_import_path, weight_file
from physverify.schema.clip import scaled_size

CODE = WeightSpec(
    WeightKind.GIT,
    "https://github.com/ByteDance-Seed/depth-anything-3.git",
    "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
)
CODE_PACKAGE_ROOT = "src"
VENDORED_PACKAGE = "depth_anything_3"
VENDORED_MODULES = ("cfg", "registry", "utils.io.input_processor", "utils.io.output_processor")
CHECKPOINT_PREFIX = "model."
PROCESS_RES_METHOD = "upper_bound_resize"
REF_VIEW_STRATEGY = "saddle_balanced"
MIN_DEPTH = 1e-6


class DepthModel(Protocol):
    def relative_depth(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Depth up to an unknown scale of one (h, w, 3) RGB frame, resized to (h, w)."""
        ...


def set_cv2_threads(threads: int) -> None:
    """Bound OpenCV's pool, which DA3's preprocessing resizes with, to ``runtime.threads``."""
    cv2.setNumThreads(threads)


def reduce_frames(frames: NDArray[np.uint8], max_side: int) -> NDArray[np.uint8]:
    """Frames whose longer side exceeds ``max_side``, area-downsampled to it; others unchanged."""
    size = scaled_size(frames.shape[2], frames.shape[1], max_side)
    if size == (frames.shape[2], frames.shape[1]):
        return frames
    return np.stack([resize_frame(frame, size) for frame in frames])


def to_disparity(depth: NDArray[np.float32]) -> NDArray[np.float32]:
    """1 / z, NaN where z is not finite or not above MIN_DEPTH."""
    valid = np.isfinite(depth) & (depth > MIN_DEPTH)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(valid, 1.0 / np.maximum(depth, MIN_DEPTH), np.nan).astype(np.float32)


def predict_disparity(model: DepthModel, frames: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Disparity (T, h, w), up to the model's unknown per-frame scale, one frame per call."""
    return to_disparity(np.stack([model.relative_depth(frame) for frame in frames]))


class TorchDA3:
    """The vendored network on one uint8 RGB frame per call: DA3's resize to ``process_res`` and
    normalisation, a forward pass under autocast, its output processing, and a bilinear resize of
    the depth back to the frame size on the CPU."""

    def __init__(
        self, net: Any, processors: tuple[Any, Any], device: str, process_res: int
    ) -> None:
        self._net = net
        self._prepare, self._collect = processors
        self._device = device
        self._process_res = process_res

    def relative_depth(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        import torch
        import torch.nn.functional as F  # noqa: N812

        images, _, _ = self._prepare(
            [frame], None, None, self._process_res, PROCESS_RES_METHOD, num_workers=1
        )
        x = images.to(self._device)[None].float()
        with torch.inference_mode(), torch.autocast(x.device.type, dtype=self._autocast_dtype()):
            raw = self._net(x, None, None, [], False, False, REF_VIEW_STRATEGY)
        z = torch.from_numpy(np.asarray(self._collect(raw).depth[0])).float()
        size = (frame.shape[0], frame.shape[1])
        resized = F.interpolate(z[None, None], size, mode="bilinear", align_corners=False)
        return np.asarray(resized[0, 0].numpy(), dtype=np.float32)

    def _autocast_dtype(self) -> Any:
        """bfloat16 where CUDA supports it, else float16, as api.py chooses."""
        import torch

        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def unprefixed(state: dict[str, Any], provider_key: str) -> dict[str, Any]:
    """The checkpoint holds the network under the ``model`` attribute of api.DepthAnything3."""
    stray = [key for key in state if not key.startswith(CHECKPOINT_PREFIX)]
    if stray:
        raise PluginError(
            f"{provider_key}: checkpoint key {stray[0]!r} lacks {CHECKPOINT_PREFIX!r}"
        )
    return {key.removeprefix(CHECKPOINT_PREFIX): value for key, value in state.items()}


def load_model(
    ctx: StageContext, checkpoint: WeightSpec, model_name: str, process_res: int, provider_key: str
) -> TorchDA3:
    """The network ``model_name`` of the checkout's config registry with ``checkpoint`` loaded
    strictly; ``provider_key`` (``stage/name``) labels a mismatch."""
    from safetensors.torch import load_file

    root = local_path(CODE, ctx.paths) / CODE_PACKAGE_ROOT
    check_vendored_package(root, VENDORED_PACKAGE, provider_key)
    with vendored_import_path(root):
        modules = [importlib.import_module(f"{VENDORED_PACKAGE}.{m}") for m in VENDORED_MODULES]
        cfg, registry, inputs, outputs = modules
        net = cfg.create_object(cfg.load_config(registry.MODEL_REGISTRY[model_name]))
    state = unprefixed(load_file(str(weight_file(checkpoint, ctx.paths))), provider_key)
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise PluginError(f"{provider_key}: checkpoint does not fit the model: {error}") from error
    processors = (inputs.InputProcessor(), outputs.OutputProcessor())
    return TorchDA3(net.to(ctx.device).eval(), processors, ctx.device, process_res)
