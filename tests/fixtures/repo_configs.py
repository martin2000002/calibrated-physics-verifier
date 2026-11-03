"""Copies of the committed ``configs/`` tree for tests that write a ``local.toml`` next to it or
edit a file of the copy, as a reviewed change to the configs would."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from physverify.config.loader import config_files

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SEALED_FILE = "verifier/sealed.toml"
CALIBRATION_FILE = "verifier/calibration.toml"


def copy_repo_configs(target: Path) -> Path:
    """Every file the loader reads from the repository's configs, subdirectories kept, into
    ``target`` (created); the root's ``local*.toml`` stay behind."""
    target.mkdir(parents=True, exist_ok=True)
    for source in config_files(REPO_CONFIGS):
        destination = target / source.relative_to(REPO_CONFIGS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, destination)
    return target


def declare(config_dir: Path, relative: str, key: str, value: str) -> None:
    """Rewrite the one ``key = ...`` line, or bracketed list, of ``config_dir/relative``."""
    path = config_dir / relative
    text = path.read_text(encoding="utf-8")
    declared, count = re.subn(rf"(?ms)^{key} = (\[.*?\]|[^\n]*)$", f"{key} = {value}", text)
    assert count == 1, f"{relative} has {count} '{key} =' lines"
    path.write_text(declared, encoding="utf-8")
