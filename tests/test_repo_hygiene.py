from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SPANISH_CODEPOINTS = (0xE1, 0xE9, 0xED, 0xF3, 0xFA, 0xF1, 0xBF, 0xA1)
SPANISH_CHARACTERS = re.compile("[" + "".join(map(chr, SPANISH_CODEPOINTS)) + "]")
TEXT_SUFFIXES = {".py", ".toml", ".md", ".sh", ".txt", ".cfg", ".yml", ".yaml", ".json"}


def _text_files(repo_root: Path) -> list[Path]:
    roots = [repo_root / name for name in ("src", "tests", "configs", "scripts", "tools")]
    files = [p for root in roots for p in root.rglob("*") if p.suffix in TEXT_SUFFIXES]
    return files + sorted(repo_root.glob("*.md"))


def test_repository_text_is_english_only(repo_root: Path) -> None:
    offenders = [
        f"{path.relative_to(repo_root)}:{number}"
        for path in _text_files(repo_root)
        if "__pycache__" not in path.parts and path.name != "local.toml"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if SPANISH_CHARACTERS.search(line)
    ]
    assert offenders == []


def test_every_package_directory_has_an_init(repo_root: Path) -> None:
    package_root = repo_root / "src" / "physverify"
    missing = [
        str(directory.relative_to(repo_root))
        for directory in [package_root, *package_root.rglob("*")]
        if directory.is_dir()
        and directory.name != "__pycache__"
        and any(directory.glob("*.py"))
        and not (directory / "__init__.py").exists()
    ]
    assert missing == []


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/clip.mp4",
        "configs/local.toml",
        ".venv/bin/python",
        "weights/sam.pth",
        "model.safetensors",
        "src/physverify/__pycache__/x.pyc",
        ".pytest_cache/v",
    ],
)
def test_local_state_is_ignored_by_git(repo_root: Path, path: str) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    result = subprocess.run([git, "-C", str(repo_root), "check-ignore", "-q", path], check=False)
    assert result.returncode == 0, f"{path} is not ignored"


@pytest.mark.parametrize("path", ["configs/local.example.toml", "configs/default.toml", "uv.lock"])
def test_shared_files_are_tracked_candidates(repo_root: Path, path: str) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    result = subprocess.run([git, "-C", str(repo_root), "check-ignore", "-q", path], check=False)
    assert result.returncode == 1, f"{path} must not be ignored"
