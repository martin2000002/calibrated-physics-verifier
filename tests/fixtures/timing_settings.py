"""The fake clip factory of the timing tests.

Shared by ``tests/timing`` and ``tests/scripts``; a conftest activates the fixtures for its
directory by importing them.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from physverify.schema.clip import ClipManifest, Split, TimeSource

FAKE_DATASET = "fake_ds"
FAKE_FPS = 16.0


def make_clip(clip_id: str, dataset: str = FAKE_DATASET, n_frames: int = 5) -> ClipManifest:
    return ClipManifest(
        dataset=dataset,
        clip_id=clip_id,
        split=Split.EVAL,
        video_path=f"{dataset}/{clip_id}.mp4",
        n_frames=n_frames,
        width=16,
        height=16,
        container_fps=FAKE_FPS,
        seconds_per_frame=1.0 / FAKE_FPS,
        time_source=TimeSource.CONTAINER_FPS,
    )


@pytest.fixture
def clip_factory() -> Callable[..., ClipManifest]:
    return make_clip
