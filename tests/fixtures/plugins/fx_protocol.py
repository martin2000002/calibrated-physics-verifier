"""Protocol used by the registry fixtures."""

from __future__ import annotations

from typing import ClassVar, Protocol


class Widget(Protocol):
    name: ClassVar[str]

    def run(self) -> str: ...
