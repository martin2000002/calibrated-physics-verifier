from pathlib import Path

import numpy as np
import pytest

from physverify.core.errors import ConfigError
from physverify.extraction.crop_encoder import (
    INPUT_PATCHES,
    PATCH_PX,
    CropEncoder,
    prepare,
    verified_checkpoint,
)


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
