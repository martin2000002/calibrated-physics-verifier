"""Contract of a video judge: generated videos -> one score each on its own scale (ADR 0013).

A judge that cannot read its own output returns ``value = None`` with a ``reason``; a missing
score is never a point on the scale.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable


@dataclass(frozen=True)
class JudgeScore:
    value: float | None
    scale: tuple[float, float]
    stamp: str
    reason: str | None = None


@dataclass(frozen=True)
class JudgeInput:
    video_id: str
    video: Path
    caption: str


@runtime_checkable
class Judge(Protocol):
    """``unavailable`` names what is missing (exit 2), or None."""

    name: ClassVar[str]

    def unavailable(self) -> str | None: ...

    def stamp(self, task: str) -> str: ...

    def judge(self, items: Sequence[JudgeInput], task: str) -> Mapping[str, JudgeScore]: ...
