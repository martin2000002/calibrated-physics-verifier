"""The reference clip the row-count windows and R4_jump's stencil reference assume is the Kubric
calibration clip of every profile: a longer or faster calibration clip has to move them too."""

from __future__ import annotations

from pathlib import Path

import pytest
from repo_configs import copy_repo_configs

from physverify.config.loader import known_profiles, load_settings, read_config_files
from physverify.physics.row_time import REFERENCE_FPS, REFERENCE_ROWS
from physverify.synth.settings import load_dataset_settings

DATASET_CONFIG = Path("datasets/kubric_synthetic.toml")


def clip_mismatches(configs: Path) -> list[str]:
    """Each profile whose calibration clip is not ``REFERENCE_ROWS`` frames at ``REFERENCE_FPS``."""
    home = {"HOME": str(configs.parent / "home")}
    found = []
    for profile in sorted(known_profiles(read_config_files(configs)[0])):
        ds = load_dataset_settings(load_settings(configs, environ=home, profile=profile))
        if (ds.n_frames, ds.fps) != (REFERENCE_ROWS, REFERENCE_FPS):
            found.append(f"{profile}: {ds.n_frames} frames at {ds.fps} fps")
    return found


@pytest.fixture
def configs(tmp_path: Path) -> Path:
    return copy_repo_configs(tmp_path / "configs")


def test_the_reference_clip_is_the_calibration_clip(configs: Path) -> None:
    assert clip_mismatches(configs) == []


@pytest.mark.parametrize(("key", "value"), [("n_frames", 48), ("fps", 30)])
def test_another_calibration_clip_is_caught(configs: Path, key: str, value: int) -> None:
    path = configs / DATASET_CONFIG
    text = path.read_text(encoding="utf-8")
    line = f"{key} = 24\n"
    assert text.count(line) == 1
    path.write_text(text.replace(line, f"{key} = {value}\n"), encoding="utf-8")
    assert len(clip_mismatches(configs)) == len(known_profiles(read_config_files(configs)[0]))
