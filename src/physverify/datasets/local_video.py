"""Hand-copied videos listed in ``<raw_dir>/local_video/<subset>/manifest.json``.

A clip is an mp4 (container rate unless ``real_fps`` is declared) or a directory of numbered
frames (``real_fps`` required). Its id is the manifest's ``clip_id``, else the file or directory
name. Queries come from the manifest, else from ``<clip_id>.queries.json`` beside it, else none.
There is no ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from physverify.core.errors import ConfigError, MissingInputError, UsageError
from physverify.core.jsonio import read_model
from physverify.core.local_manifest import (
    MANIFEST_FILE,
    ClipEntry,
    LocalManifest,
    QueryList,
    clip_id_of,
    to_queries,
)
from physverify.core.models import FrozenModel
from physverify.core.video import FrameDirVideoSource, Mp4VideoSource
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.schema.clip import ClipManifest, ObjectQuery, Split, TimeSource

if TYPE_CHECKING:
    from physverify.config.models import Settings

QUERIES_SUFFIX = ".queries.json"
DEFAULT_SUBSET = "default"
NAME = "local_video"


class _QueryFile(FrozenModel):
    queries: QueryList


class LocalVideoDataset:
    name: ClassVar[str] = NAME

    def __init__(self, settings: Settings) -> None:
        self._root = settings.paths.raw_dir / self.name
        self._max_frame_bytes = settings.runtime.max_frame_bytes

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        directory = self._root / (subset or DEFAULT_SUBSET)
        path = directory / MANIFEST_FILE
        if not path.is_file():
            raise MissingInputError(f"no {MANIFEST_FILE} in {directory} (it lists the clips)")
        clips = [
            self._manifest(directory, entry) for entry in read_model(LocalManifest, path).clips
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
            return to_queries(read_model(_QueryFile, sidecar).queries)
        return ()


PLUGINS = (LocalVideoDataset,)
