"""Settings whose data directory is ``tmp_path / "data"``; a conftest activates the fixtures for
its directory by importing them."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def settings_environ(environ: dict[str, str], data_dir: Path) -> dict[str, str]:
    return {**environ, "PHYSVERIFY__PATHS__DATA_DIR": str(data_dir)}
