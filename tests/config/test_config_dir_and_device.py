"""The default config directory and the device names the CLI accepts."""

from __future__ import annotations

from pathlib import Path

import pytest

from physverify.config.loader import CONFIG_DIR_ENV, default_config_dir
from physverify.config.models import is_device

REPO = Path(__file__).resolve().parents[2]


def test_the_config_dir_defaults_to_the_repository_configs() -> None:
    assert default_config_dir({}) == REPO / "configs"


def test_the_environment_overrides_the_config_dir(tmp_path: Path) -> None:
    assert default_config_dir({CONFIG_DIR_ENV: str(tmp_path)}) == tmp_path


@pytest.mark.parametrize(
    ("value", "valid"),
    [("cpu", True), ("cuda", True), ("cuda:3", True), ("gpu", False), ("cuda:", False)],
)
def test_device_names(value: str, valid: bool) -> None:
    assert is_device(value) is valid
