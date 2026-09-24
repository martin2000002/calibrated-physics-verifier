"""AC15: hand-copied videos and frame directories listed in a local manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from conftest import SettingsFactory
from physverify.core.errors import ConfigError, MissingInputError
from physverify.datasets.local_video import MANIFEST_FILE, QUERIES_SUFFIX, LocalVideoDataset
from physverify.schema.clip import ObjectQuery, TimeSource

SUBSET = "smoke"
SIZE = (40, 32)
N_FRAMES = 5
CONTAINER_FPS = 10.0


def frames() -> list[np.ndarray]:
    return [np.full((SIZE[1], SIZE[0], 3), 40 * i, np.uint8) for i in range(N_FRAMES)]


def write_mp4(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), CONTAINER_FPS, SIZE)
    for frame in frames():
        writer.write(frame)
    writer.release()


def write_frame_dir(path: Path) -> None:
    path.mkdir()
    for i, frame in enumerate(frames()):
        cv2.imwrite(str(path / f"rgba_{i:05d}.png"), frame)


def subset_dir(make_settings: SettingsFactory) -> tuple[LocalVideoDataset, Path]:
    settings = make_settings()
    directory = settings.paths.raw_dir / "local_video" / SUBSET
    directory.mkdir(parents=True)
    return LocalVideoDataset(settings), directory


def write_manifest(directory: Path, clips: list[dict[str, Any]]) -> None:
    (directory / MANIFEST_FILE).write_text(json.dumps({"clips": clips}))


def test_mp4_and_frame_directory_clips(make_settings: SettingsFactory) -> None:
    dataset, directory = subset_dir(make_settings)
    write_mp4(directory / "drop.mp4")
    write_frame_dir(directory / "kubric")
    (directory / f"drop{QUERIES_SUFFIX}").write_text(json.dumps({"queries": ["cup"]}))
    write_manifest(
        directory,
        [
            {"video": "drop.mp4"},
            {
                "video": "kubric/",
                "real_fps": 24,
                "queries": ["ball", {"phrase": "cube", "expected_count": 2}],
            },
        ],
    )
    mp4, frame_dir = dataset.clips(SUBSET)
    assert (mp4.clip_id, frame_dir.clip_id) == ("drop", "kubric")
    assert mp4.object_queries == (ObjectQuery("cup"),)
    assert frame_dir.object_queries == (ObjectQuery("ball"), ObjectQuery("cube", 2))
    assert mp4.time_source is TimeSource.CONTAINER_FPS
    assert mp4.seconds_per_frame == pytest.approx(1.0 / CONTAINER_FPS)
    assert frame_dir.time_source is TimeSource.DECLARED
    assert frame_dir.seconds_per_frame == pytest.approx(1.0 / 24)
    for clip in (mp4, frame_dir):
        assert (clip.n_frames, clip.width, clip.height) == (N_FRAMES, *SIZE)
        assert dataset.ground_truth(clip) is None
        assert dataset.video(clip).native_size == SIZE


def test_declared_rate_overrides_the_container_and_queries_may_be_absent(
    make_settings: SettingsFactory,
) -> None:
    dataset, directory = subset_dir(make_settings)
    write_mp4(directory / "drop.mp4")
    write_manifest(directory, [{"video": "drop.mp4", "real_fps": 240}])
    (clip,) = dataset.clips(SUBSET)
    assert clip.time_source is TimeSource.DECLARED
    assert clip.container_fps == pytest.approx(CONTAINER_FPS)
    assert clip.seconds_per_frame == pytest.approx(1.0 / 240)
    assert clip.object_queries == ()


def test_a_frame_directory_needs_a_declared_rate(make_settings: SettingsFactory) -> None:
    dataset, directory = subset_dir(make_settings)
    write_frame_dir(directory / "kubric")
    write_manifest(directory, [{"video": "kubric/"}])
    with pytest.raises(ConfigError, match="real_fps"):
        dataset.clips(SUBSET)


def test_missing_or_invalid_manifests(make_settings: SettingsFactory) -> None:
    dataset, directory = subset_dir(make_settings)
    with pytest.raises(MissingInputError, match=MANIFEST_FILE):
        dataset.clips(SUBSET)
    write_manifest(directory, [{"video": "drop.mp4", "fps": 3}])
    with pytest.raises(ConfigError, match="invalid"):
        dataset.clips(SUBSET)
