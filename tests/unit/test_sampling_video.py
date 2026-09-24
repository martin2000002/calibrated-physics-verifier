from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from physverify.config.models import RuntimeSettings
from physverify.core.errors import ResourceBudgetError
from physverify.core.resample import scale_intrinsics
from physverify.core.video import FrameDirVideoSource, check_frame_budget
from physverify.schema.clip import (
    ClipManifest,
    SamplingSpec,
    Split,
    TimeSource,
    resolve_sampling,
    scaled_size,
)


def _clip_4k() -> ClipManifest:
    return ClipManifest(
        dataset="physics_iq",
        clip_id="c",
        split=Split.EVAL,
        video_path="c.mp4",
        n_frames=150,
        width=3840,
        height=2160,
        container_fps=30.0,
        seconds_per_frame=1.0 / 30.0,
        time_source=TimeSource.CONTAINER_FPS,
    )


def test_resolve_sampling_4k_to_640() -> None:
    sampling = resolve_sampling(_clip_4k(), SamplingSpec(target_fps=8.0, max_side=640))
    assert (sampling.width, sampling.height) == (640, 360)
    assert sampling.step == 4
    assert sampling.frame_indices.tolist() == list(range(0, 150, 4))
    assert sampling.seconds_per_frame == pytest.approx(1 / 30.0, abs=1e-15)
    assert np.diff(sampling.times()) == pytest.approx(np.full(37, 4 / 30.0), abs=1e-12)


def test_intrinsics_scale_by_one_sixth() -> None:
    k = np.array([[3000.0, 0.0, 1920.0], [0.0, 3000.0, 1080.0], [0.0, 0.0, 1.0]])
    scaled = scale_intrinsics(k, (3840, 2160), (640, 360))
    expected = k.copy()
    expected[:2] /= 6.0
    np.testing.assert_allclose(scaled, expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    ("size", "max_side", "expected"),
    [
        ((1280, 720), None, (1280, 720)),
        ((1280, 720), 2000, (1280, 720)),
        ((720, 1280), 640, (360, 640)),
    ],
)
def test_scaled_size(
    size: tuple[int, int], max_side: int | None, expected: tuple[int, int]
) -> None:
    assert scaled_size(*size, max_side) == expected


def test_frame_directory_source(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for index in range(5):
        frame = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{index:05d}.png"), frame)
    source = FrameDirVideoSource(
        tmp_path, real_fps=24.0, max_frame_bytes=RuntimeSettings().max_frame_bytes
    )
    assert source.n_frames == 5 and source.native_size == (64, 48)
    frames = source.frames(0, 5, 2, max_side=32)
    assert frames.shape == (3, 24, 32, 3) and frames.dtype == np.uint8
    full = source.frames(1, 2)
    expected = cv2.cvtColor(cv2.imread(str(tmp_path / "00001.png")), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(full[0], expected)


def test_budget_refuses_240_native_4k_frames() -> None:
    budget = RuntimeSettings().max_frame_bytes
    with pytest.raises(ResourceBudgetError, match="max_frame_bytes"):
        check_frame_budget(240, (3840, 2160), budget)
    check_frame_budget(240, (640, 360), budget)
