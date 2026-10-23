"""An autouse guard that a conftest activates for its directory by importing it."""

from __future__ import annotations

import socket
from typing import Any

import pytest


def _refuse(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("tests must not touch the network")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
