"""Shared fixtures. Tests write only under ``tmp_path``; configs are copied there."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from cli_runner import CliResult, CliRunner
from repo_configs import copy_repo_configs
from settings_factory import SettingsFactory

from physverify.cli.main import run
from physverify.config.loader import load_settings
from physverify.config.models import Settings

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
def settings_environ(environ: dict[str, str]) -> dict[str, str]:
    """The environment ``make_settings`` loads with; a package overrides it to move its data."""
    return environ


@pytest.fixture
def settings_overrides() -> tuple[str, ...]:
    """Overrides ``make_settings`` applies before the caller's; a package overrides it."""
    return ()


@pytest.fixture
def make_settings(
    config_dir: Path, settings_environ: dict[str, str], settings_overrides: tuple[str, ...]
) -> SettingsFactory:
    def build(*overrides: str, profile: str | None = None) -> Settings:
        return load_settings(
            config_dir,
            environ=settings_environ,
            profile=profile,
            overrides=(*settings_overrides, *overrides),
        )

    return build


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
