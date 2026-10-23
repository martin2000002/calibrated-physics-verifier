"""Copies of the committed ``configs/`` tree for tests that write a ``local.toml`` next to it."""

from __future__ import annotations

import shutil
from pathlib import Path

from physverify.config.loader import config_files

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def copy_repo_configs(target: Path) -> Path:
    """Every file the loader reads from the repository's configs, subdirectories kept, into
    ``target`` (created); the root's ``local*.toml`` stay behind."""
    target.mkdir(parents=True, exist_ok=True)
    for source in config_files(REPO_CONFIGS):
        destination = target / source.relative_to(REPO_CONFIGS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, destination)
    return target
