"""The ``manifest.json`` of a hand-copied ``local_video`` subset and its object queries.

Below the datasets layer so the producers of such subsets (generation, sizing, fine-tuning) and
the dataset that reads them share one format."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from physverify.core.models import FrozenModel
from physverify.schema.clip import ObjectQuery

MANIFEST_FILE = "manifest.json"


class QueryEntry(FrozenModel):
    phrase: str = Field(min_length=1)
    expected_count: int | None = Field(default=None, ge=1)


QueryList = list[str | QueryEntry]


class ClipEntry(FrozenModel):
    video: str = Field(min_length=1)
    clip_id: str | None = Field(default=None, min_length=1)
    real_fps: float | None = Field(default=None, gt=0)
    queries: QueryList | None = None


class LocalManifest(FrozenModel):
    clips: list[ClipEntry]


def to_queries(entries: QueryList) -> tuple[ObjectQuery, ...]:
    return tuple(
        ObjectQuery(phrase=e) if isinstance(e, str) else ObjectQuery(e.phrase, e.expected_count)
        for e in entries
    )


def clip_id_of(entry: ClipEntry) -> str:
    if entry.clip_id is not None:
        return entry.clip_id
    path = Path(entry.video)
    return path.name if entry.video.endswith("/") else path.stem
