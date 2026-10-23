"""Shared fixtures. Tests write only under ``tmp_path``; configs are copied there."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from cli_runner import CliResult, CliRunner
from repo_configs import copy_repo_configs

from physverify.cli.main import run

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / "locks" / "heavy.lock"


@pytest.fixture
def config_dir(tmp_path: Path, lock_path: Path) -> Path:
    """The committed configs plus a local.toml whose heavy lock lives in ``tmp_path``."""
    target = copy_repo_configs(tmp_path / "repo" / "configs")
    (target / "local.toml").write_text(f'[runtime]\nheavy_lock = "{lock_path}"\n', encoding="utf-8")
    return target


@pytest.fixture
def environ(tmp_path: Path) -> dict[str, str]:
    return {"HOME": str(tmp_path / "home"), "PATH": os.environ.get("PATH", "")}


@pytest.fixture
def cli(config_dir: Path, environ: dict[str, str], capsys: pytest.CaptureFixture[str]) -> CliRunner:
    def invoke(
        *argv: str,
        env: Mapping[str, str] | None = None,
        commands_package: str = "physverify.cli.commands",
    ) -> CliResult:
        args: Sequence[str] = ("--config-dir", str(config_dir), *argv)
        code = run(args, {**environ, **(env or {})}, commands_package=commands_package)
        captured = capsys.readouterr()
        return CliResult(code, captured.out, captured.err)

    return invoke
