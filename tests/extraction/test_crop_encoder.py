from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from physverify.core.errors import ConfigError
from physverify.extraction import crop_encoder
from physverify.extraction.crop_encoder import (
    INPUT_PATCHES,
    PATCH_PX,
    CropEncoder,
    prepare,
    vda_crop_encoder,
    verified_checkpoint,
)
from physverify.extraction.providers import _vda
from physverify.extraction.providers._vda import CHECKPOINT
from physverify.extraction.weights import weight_file


def test_prepare_resizes_to_whole_patches_and_normalises() -> None:
    crops = np.full((2, 64, 64, 3), 255.0)
    batch = prepare(crops)
    side = INPUT_PATCHES * PATCH_PX
    assert batch.shape == (2, 3, side, side)
    assert np.allclose(batch[0, 0], (1.0 - 0.485) / 0.229)


def test_encoder_runs_the_injected_backbone() -> None:
    encoder = CropEncoder(lambda batch: batch.mean(axis=(2, 3)), "abc")
    tokens = encoder.encode(np.zeros((3, 64, 64, 3)))
    assert tokens.shape == (3, 3)


def test_checkpoint_with_another_hash_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "weights.pth"
    path.write_bytes(b"not the checkpoint")
    with pytest.raises(ConfigError):
        verified_checkpoint(path)


def test_the_configured_encoder_is_refused_before_loading_on_another_hash(
    make_settings: Any,
) -> None:
    paths = make_settings().paths
    checkpoint = weight_file(CHECKPOINT, paths)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"not the checkpoint")
    with pytest.raises(ConfigError, match="expected"):
        vda_crop_encoder(paths)


def test_the_encoder_and_the_depth_provider_share_one_loader(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, ...]] = []

    def build(*args: Any, **kwargs: Any) -> Any:
        calls.append((*args, kwargs))
        return SimpleNamespace(pretrained=None)

    paths = make_settings().paths
    monkeypatch.setattr(crop_encoder, "build_model", build)
    monkeypatch.setattr(crop_encoder, "verified_checkpoint", lambda path: "sha")
    monkeypatch.setattr(_vda, "build_model", build)
    assert vda_crop_encoder(paths, "cpu").sha256 == "sha"
    _vda.load_model(SimpleNamespace(paths=paths, device="cpu"))  # type: ignore[arg-type]
    assert calls == [
        (paths, "cpu", {"provider": "crop_encoder/vda_small"}),
        (paths, "cpu", CHECKPOINT, False, "depth/vda_small", _vda.SMALL_ENCODER, {}),
    ]
