"""PISA real drops (ADR 0007 §4.2, §5.6): ``movie.mp4`` and ``clip_info.json`` of selected clips,
read by HTTP range from ``pisabench/real.zip`` at a pinned revision. There is no ground truth;
the capture rate ``real_fps`` is declared, not measured.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import zipfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from pydantic import Field

from physverify.config.models import SIZE_PRIORS_SECTION, SizePriorTable
from physverify.core.atomic import write_text_atomic
from physverify.core.digests import file_sha256
from physverify.core.download import (
    MANIFEST_FILE,
    FetchedFile,
    FetchManifest,
    extract_member,
    intact,
    read_manifest,
    remote_zip,
    retrying,
    write_manifest,
)
from physverify.core.errors import MissingInputError, UsageError
from physverify.core.log import get_logger
from physverify.core.models import FrozenModel
from physverify.core.video import Mp4VideoSource
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.datasets._layout import load_annotations, relative_to_data, resolve_data_path
from physverify.datasets._queries import derive_queries
from physverify.schema.clip import ClipManifest, ObjectQuery, Split, TimeSource
from physverify.sizing.priors import lookup_label

if TYPE_CHECKING:
    from physverify.config.models import Settings

SECTION = "pisa_real"
LOCAL = "local"
FULL = "full"
SUBSETS = (LOCAL, FULL)
ARCHIVE_ROOT = "real"
MOVIE = "movie.mp4"
CLIP_INFO = "clip_info.json"
INDEX_FILE = "index.json"
FETCH_HINT = "physverify data fetch pisa_real"
CLIP_INFO_PATTERN = re.compile(rf"^{ARCHIVE_ROOT}/([^/]+)/{re.escape(CLIP_INFO)}$")

log = get_logger("datasets.pisa_real")
ArchiveOpener = Callable[[str], zipfile.ZipFile]


class PisaRealSettings(FrozenModel):
    repo_url: str
    revision: str
    archive: str
    license: str
    real_fps: float = Field(gt=0)
    max_clips: int = Field(ge=0)
    max_frames: int = Field(ge=0)
    seed: int
    local_ids: tuple[str, ...]
    fall_verbs: tuple[str, ...]
    retries: int = Field(ge=0)


def select_ids(
    prompts: Mapping[str, str], n: int, seed: int, has_prior: Callable[[str], bool]
) -> tuple[str, ...]:
    """A seeded shuffle of the ids, those whose prompt has a size prior first; ``n`` = 0 is all."""
    ids = sorted(prompts)
    order = np.random.default_rng(seed).permutation(len(ids))
    shuffled = [ids[i] for i in order]
    ranked = sorted(shuffled, key=lambda clip_id: not has_prior(prompts[clip_id]))
    return tuple(ranked[: n or None])


def read_prompts(archive: zipfile.ZipFile) -> dict[str, str]:
    """``{id: prompt}`` from every ``real/<id>/clip_info.json`` of the archive."""
    prompts: dict[str, str] = {}
    for name in archive.namelist():
        match = CLIP_INFO_PATTERN.match(name)
        if match is not None:
            info = json.loads(archive.read(name))
            prompts[match.group(1)] = str(info.get("prompt", ""))
    return prompts


class _Archive:
    """The remote zip, opened on first use and closed with the fetch."""

    def __init__(self, url: str, opener: ArchiveOpener, stack: ExitStack, retries: int) -> None:
        self._url, self._opener, self._stack, self._retries = url, opener, stack, retries
        self._zip: zipfile.ZipFile | None = None

    def get(self) -> zipfile.ZipFile:
        if self._zip is None:
            opened = retrying(lambda: self._opener(self._url), self._retries, f"open {self._url}")
            self._zip = self._stack.enter_context(opened)
        return self._zip


class PisaRealDataset:
    name: ClassVar[str] = "pisa_real"

    def __init__(self, settings: Settings, opener: ArchiveOpener | None = None) -> None:
        self._s = settings.section(SECTION, PisaRealSettings)
        self._priors = settings.section(SIZE_PRIORS_SECTION, SizePriorTable).root
        paths = settings.paths
        self._root = paths.raw_dir / self.name
        self._data_dir = paths.data_dir
        self._annotations = paths.annotations_dir / f"{self.name}.json"
        self._max_frame_bytes = settings.runtime.max_frame_bytes
        self._opener = opener or remote_zip

    @property
    def archive_url(self) -> str:
        s = self._s
        return f"{s.repo_url}/resolve/{s.revision}/{s.archive}"

    def has_prior(self, prompt: str) -> bool:
        return any(
            lookup_label(q.phrase, self._priors) is not None
            for q in derive_queries(prompt, self._s.fall_verbs)
        )

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        ids = self._wanted(subset or LOCAL, self._root, None)
        queries = load_annotations(self._annotations)
        return [self._manifest(clip_id, queries) for clip_id in ids]

    def video(self, clip: ClipManifest) -> VideoSource:
        path = resolve_data_path(clip.video_path, self._data_dir)
        return Mp4VideoSource(path, self._max_frame_bytes, real_fps=clip.real_fps)

    def ground_truth(self, clip: ClipManifest) -> GroundTruthStore | None:
        return None

    def fetch(self, subset: str | None, dest: Path) -> None:
        subset = subset or LOCAL
        manifest_path = dest / MANIFEST_FILE
        previous = read_manifest(manifest_path)
        known = {} if previous is None else {f.relative: f for f in previous.files}
        with ExitStack() as stack:
            archive = _Archive(self.archive_url, self._opener, stack, self._s.retries)
            ids = self._wanted(subset, dest, archive)
            files = [
                self._fetch_member(archive, dest, clip_id, member, known)
                for clip_id in ids
                for member in (MOVIE, CLIP_INFO)
            ]
        log.info("pisa_real ids: %s", ", ".join(ids))
        write_manifest(
            manifest_path,
            FetchManifest(
                dataset=self.name,
                source=self._s.repo_url,
                revision=self._s.revision,
                license=self._s.license,
                subset=subset,
                fetched_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                files=tuple(files),
                clips=ids,
            ),
        )

    def _wanted(self, subset: str, root: Path, archive: _Archive | None) -> tuple[str, ...]:
        """Pinned ``local_ids`` for the local subset, else a seeded selection over the index."""
        if subset not in SUBSETS:
            raise UsageError(f"unknown pisa_real subset {subset!r}; known: {', '.join(SUBSETS)}")
        if subset == LOCAL and self._s.local_ids:
            return self._s.local_ids
        prompts = self._index(root, archive)
        return select_ids(prompts, self._s.max_clips, self._s.seed, self.has_prior)

    def _index(self, root: Path, archive: _Archive | None) -> dict[str, str]:
        path = root / INDEX_FILE
        if path.is_file():
            index: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
            return index
        if archive is None:
            raise MissingInputError(f"{path} is missing; run `{FETCH_HINT}`")
        prompts = read_prompts(archive.get())
        write_text_atomic(path, json.dumps(prompts, indent=2, sort_keys=True) + "\n")
        return prompts

    def _fetch_member(
        self,
        archive: _Archive,
        dest: Path,
        clip_id: str,
        member: str,
        known: Mapping[str, FetchedFile],
    ) -> FetchedFile:
        relative = f"{clip_id}/{member}"
        target = dest / relative
        record = known.get(relative)
        if record is not None and intact(target, record):
            return record
        size = extract_member(archive.get(), f"{ARCHIVE_ROOT}/{relative}", target)
        log.info("%s: %d bytes", relative, size)
        return FetchedFile(relative=relative, size=size, sha256=file_sha256(target))

    def _manifest(
        self, clip_id: str, queries: Mapping[str, tuple[ObjectQuery, ...]]
    ) -> ClipManifest:
        movie, info = self._root / clip_id / MOVIE, self._root / clip_id / CLIP_INFO
        if not movie.is_file() or not info.is_file():
            raise MissingInputError(f"pisa_real clip {clip_id} is missing; run `{FETCH_HINT}`")
        prompt = str(json.loads(info.read_text(encoding="utf-8")).get("prompt", ""))
        source = Mp4VideoSource(movie, self._max_frame_bytes)
        width, height = source.native_size
        cap = self._s.max_frames or source.n_frames
        return ClipManifest(
            dataset=self.name,
            clip_id=clip_id,
            split=Split.EVAL,
            video_path=relative_to_data(movie, self._data_dir),
            n_frames=min(source.n_frames, cap),
            width=width,
            height=height,
            container_fps=source.real_fps,
            seconds_per_frame=1.0 / self._s.real_fps,
            time_source=TimeSource.DECLARED,
            object_queries=queries.get(clip_id) or derive_queries(prompt, self._s.fall_verbs),
            prompt=prompt,
            license=self._s.license,
        )


PLUGINS = (PisaRealDataset,)
