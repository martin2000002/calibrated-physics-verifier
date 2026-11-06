"""Frozen crop encoder: the ViT-S backbone of Video Depth Anything Small, identified by the sha256
of its checkpoint file, embeds RGB crops as the class token of its last layer.

Crops are resized to a whole number of 14 px patches and normalised with the ImageNet statistics
the backbone was trained with. The backbone is injectable so the image path is testable without
weights.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from physverify.config.models import PathsSettings
from physverify.core.digests import file_sha256
from physverify.core.errors import ConfigError
from physverify.extraction.contact_crops import nearest_indices
from physverify.extraction.providers._vda import CHECKPOINT, build_model
from physverify.extraction.weights import weight_file

FloatArray = NDArray[np.float64]
Backbone = Callable[[FloatArray], FloatArray]

PATCH_PX = 14
INPUT_PATCHES = 5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


@dataclass(frozen=True)
class CropEncoder:
    backbone: Backbone
    sha256: str

    def encode(self, crops: FloatArray) -> FloatArray:
        """(n, s, s, 3) RGB crops in [0, 255] -> (n, d) tokens."""
        if crops.shape[0] == 0:
            return np.zeros((0, 0))
        return np.asarray(self.backbone(prepare(crops)), dtype=np.float64)


def prepare(crops: FloatArray) -> FloatArray:
    """(n, s, s, 3) in [0, 255] -> (n, 3, P, P) normalised, P = INPUT_PATCHES * PATCH_PX."""
    side = INPUT_PATCHES * PATCH_PX
    index = nearest_indices(side, crops.shape[1])
    resized = crops[:, index][:, :, index] / 255.0
    normalised = (resized - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalised.transpose(0, 3, 1, 2))


def verified_checkpoint(path: Path) -> str:
    digest = file_sha256(path)
    if digest != CHECKPOINT.sha256:
        raise ConfigError(f"crop encoder: {path} has sha256 {digest}, expected {CHECKPOINT.sha256}")
    return digest


def vda_backbone(model: object) -> Backbone:
    """Class token of the last layer of ``model.pretrained`` (a DINOv2 ViT), on CPU, no grad."""
    import torch

    vit = model.pretrained  # type: ignore[attr-defined]

    def run(batch: FloatArray) -> FloatArray:
        with torch.no_grad():
            tensor = torch.from_numpy(batch).float()
            _, cls = vit.get_intermediate_layers(tensor, 1, return_class_token=True)[0]
            return np.asarray(cls.cpu().numpy(), dtype=np.float64)

    return run


def vda_crop_encoder(paths: PathsSettings, device: str = "cpu") -> CropEncoder:
    """The VDA-S crop encoder from the configured weights, refused unless its hash matches."""
    sha = verified_checkpoint(weight_file(CHECKPOINT, paths))
    return CropEncoder(
        vda_backbone(build_model(paths, device, provider="crop_encoder/vda_small")), sha
    )
