"""Selection records of ``objectron_static`` and the preregistered candidate order."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import Field

from physverify.core.digests import text_sha256
from physverify.core.models import FrozenModel
from physverify.sizing import objectron

TAKEN, TOO_LARGE, FAILED = "taken", "too_large", "failed"


class Candidate(FrozenModel):
    video_id: str
    status: str
    size: int | None = None


class StaticClip(FrozenModel):
    clip_id: str
    video_id: str
    label: str
    fps: float = Field(gt=0)
    n_frames: int = Field(ge=1)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class StaticSelection(FrozenModel):
    """What a fetch chose and why; written once, after the whole selection."""

    seed: int
    split: str
    per_class: int
    max_video_mb: float
    seconds: float
    excluded: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    clips: tuple[StaticClip, ...]

    def rule(self) -> tuple[Any, ...]:
        return (self.seed, self.split, self.per_class, self.max_video_mb, self.seconds)


def order_key(seed: int, video_id: str) -> str:
    return text_sha256(f"{seed}:{video_id}")


def ordered(
    refs: Iterable[objectron.VideoRef], excluded: frozenset[str], seed: int
) -> list[objectron.VideoRef]:
    """The candidates of one class in the preregistered order."""
    kept = (r for r in refs if r.video_id not in excluded)
    return sorted(kept, key=lambda r: order_key(seed, r.video_id))


def read_selection(path: Path) -> StaticSelection | None:
    if not path.is_file():
        return None
    return StaticSelection.model_validate_json(path.read_text(encoding="utf-8"))
