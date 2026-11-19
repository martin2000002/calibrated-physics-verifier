"""Result type of the ``cli`` fixture (see conftest.py)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class CliResult:
    code: int
    out: str
    err: str


CliRunner = Callable[..., CliResult]
