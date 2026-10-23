"""Timing settings over ``configs/`` with ``tmp_path`` as data, and the fake clip factory.

Shared by ``tests/timing`` and ``tests/scripts``; a conftest activates the fixtures for its
directory by importing them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.schema.clip import ClipManifest, Split, TimeSource

FAKE_DATASET = "fake_ds"
FAKE_FPS = 16.0


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def timing_environ(environ: dict[str, str], data_dir: Path) -> dict[str, str]:
    return {**environ, "PHYSVERIFY__PATHS__DATA_DIR": str(data_dir)}


@pytest.fixture
def make_settings(config_dir: Path, timing_environ: dict[str, str]) -> Callable[..., Settings]:
    """``make_settings("section.key=value", ..., profile=None)`` over the committed configs."""

    def build(*overrides: str, profile: str | None = None) -> Settings:
        return load_settings(
            config_dir, environ=timing_environ, profile=profile, overrides=overrides
        )

    return build


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
