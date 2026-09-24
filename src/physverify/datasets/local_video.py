"""Hand-copied videos listed in ``<raw_dir>/local_video/<subset>/manifest.json`` (ADR 0005, 3.11).

A clip is an mp4 (container rate unless ``real_fps`` is declared) or a directory of numbered
frames (``real_fps`` required). Queries come from the manifest, else from
``<clip_id>.queries.json`` beside it, else none. There is no ground truth.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from physverify.core.errors import ConfigError, MissingInputError, UsageError
from physverify.core.video import FrameDirVideoSource, Mp4VideoSource
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.schema.clip import ClipManifest, ObjectQuery, Split, TimeSource

if TYPE_CHECKING:
    from physverify.config.models import Settings

MANIFEST_FILE = "manifest.json"
QUERIES_SUFFIX = ".queries.json"
DEFAULT_SUBSET = "default"


class QueryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    phrase: str = Field(min_length=1)
    expected_count: int | None = Field(default=None, ge=1)


QueryList = list[str | QueryEntry]


class ClipEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    video: str = Field(min_length=1)
    real_fps: float | None = Field(default=None, gt=0)
    queries: QueryList | None = None


class LocalManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    clips: list[ClipEntry]


def to_queries(entries: QueryList) -> tuple[ObjectQuery, ...]:
    return tuple(
        ObjectQuery(phrase=e) if isinstance(e, str) else ObjectQuery(e.phrase, e.expected_count)
        for e in entries
    )


def clip_id_of(entry: ClipEntry) -> str:
    path = Path(entry.video)
    return path.name if entry.video.endswith("/") else path.stem


def _read_model[M: BaseModel](path: Path, model: type[M]) -> M:
    try:
        return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ConfigError(f"{path} is invalid:\n{exc}") from exc


class _QueryFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    queries: QueryList


class LocalVideoDataset:
    name: ClassVar[str] = "local_video"

    def __init__(self, settings: Settings) -> None:
        self._root = settings.paths.raw_dir / self.name
        self._max_frame_bytes = settings.runtime.max_frame_bytes

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        directory = self._root / (subset or DEFAULT_SUBSET)
        path = directory / MANIFEST_FILE
        if not path.is_file():
            raise MissingInputError(f"no {MANIFEST_FILE} in {directory} (see ADR 0005, 3.11)")
        clips = [
            self._manifest(directory, entry) for entry in _read_model(path, LocalManifest).clips
        ]
        ids = [clip.clip_id for clip in clips]
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        if duplicated:
            raise ConfigError(f"{path}: duplicated clip ids {duplicated}")
        return clips

    def video(self, clip: ClipManifest) -> VideoSource:
        path = Path(clip.video_path)
        if path.is_dir():
            return FrameDirVideoSource(path, clip.real_fps, self._max_frame_bytes)
        return Mp4VideoSource(path, self._max_frame_bytes, real_fps=clip.real_fps)

    def ground_truth(self, clip: ClipManifest) -> GroundTruthStore | None:
        return None

    def fetch(self, subset: str | None, dest: Path) -> None:
        raise UsageError(
            f"local videos are copied by hand into {self._root}/<subset>/ with a {MANIFEST_FILE}"
        )

    def _manifest(self, directory: Path, entry: ClipEntry) -> ClipManifest:
        path = directory / entry.video
        clip_id = clip_id_of(entry)
        source, container_fps = self._open(path, entry.real_fps)
        real_fps = entry.real_fps or container_fps
        width, height = source.native_size
        return ClipManifest(
            dataset=self.name,
            clip_id=clip_id,
            split=Split.EVAL,
            video_path=str(path.resolve()),
            n_frames=source.n_frames,
            width=width,
            height=height,
            container_fps=container_fps,
            seconds_per_frame=1.0 / real_fps,
            time_source=TimeSource.CONTAINER_FPS if entry.real_fps is None else TimeSource.DECLARED,
            object_queries=self._queries(directory, clip_id, entry),
        )

    def _open(self, path: Path, real_fps: float | None) -> tuple[VideoSource, float]:
        """The source and its container rate (a frame directory's rate is the declared one)."""
        if path.is_dir():
            if real_fps is None:
                raise ConfigError(f"{path}: a frame directory needs real_fps in {MANIFEST_FILE}")
            return FrameDirVideoSource(path, real_fps, self._max_frame_bytes), real_fps
        source = Mp4VideoSource(path, self._max_frame_bytes)
        if not source.real_fps > 0:
            raise ConfigError(f"{path}: the container declares no frame rate; set real_fps")
        return source, source.real_fps

    @staticmethod
    def _queries(directory: Path, clip_id: str, entry: ClipEntry) -> tuple[ObjectQuery, ...]:
        if entry.queries is not None:
            return to_queries(entry.queries)
        sidecar = directory / f"{clip_id}{QUERIES_SUFFIX}"
        if sidecar.is_file():
            return to_queries(_read_model(sidecar, _QueryFile).queries)
        return ()


PLUGINS = (LocalVideoDataset,)
