"""Extraction test fixtures: no network, the analytic clip, fakes and settings builders."""

from __future__ import annotations

import socket
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

# Shared helpers (fakes.py, support.py) live next to the tests; importlib mode does not add
# this directory to sys.path by itself.
HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fakes import FAKES, JOURNAL  # noqa: E402
from synthetic_clip import SyntheticClip, SyntheticFixtureDataset, build_clip  # noqa: E402

from physverify.cli.context import PLUGIN_KINDS  # noqa: E402
from physverify.config.loader import load_settings  # noqa: E402
from physverify.config.models import RuntimeSettings, Settings  # noqa: E402
from physverify.extraction._base import StageProvider, provider_key  # noqa: E402

SettingsFactory = Callable[..., Settings]


def _refuse(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("extraction tests must not touch the network")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)


@pytest.fixture(autouse=True)
def journal() -> Iterator[None]:
    JOURNAL.reset()
    yield
    JOURNAL.reset()


@pytest.fixture(scope="session")
def real_providers() -> Mapping[str, type[StageProvider]]:
    return dict(PLUGIN_KINDS["providers"]())


@pytest.fixture
def providers(real_providers: Mapping[str, type[StageProvider]]) -> dict[str, type[StageProvider]]:
    """The real registry plus ``<stage>/fake`` for DETECT, TRACK and DEPTH."""
    return {**real_providers, **{provider_key(fake): fake for fake in FAKES}}


@pytest.fixture
def make_settings(config_dir: Path, environ: dict[str, str]) -> SettingsFactory:
    def build(*overrides: str, profile: str | None = None) -> Settings:
        return load_settings(config_dir, environ=environ, profile=profile, overrides=overrides)

    return build


@pytest.fixture(scope="session")
def fixture_clip() -> SyntheticClip:
    return build_clip(RuntimeSettings())


@pytest.fixture
def fixture_dataset() -> SyntheticFixtureDataset:
    return SyntheticFixtureDataset()
