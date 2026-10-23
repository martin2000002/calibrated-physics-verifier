"""Physics-IQ real videos (ADR 0007 §4.2, §5.6): the ``testing`` part of each scenario and its
pixel-change mask, downloaded from the public GCS bucket. There is no per-stage ground truth.
The plugin reads the configured ``view``; a subclass reads other views (``views_of``) of the same
scenarios, takes and annotations under its own name and raw directory.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from physverify.config.models import FrozenModel
from physverify.core.errors import MissingInputError, UsageError
from physverify.core.log import get_logger
from physverify.core.video import Mp4VideoSource
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.datasets._download import (
    MANIFEST_FILE,
    FetchedFile,
    FetchManifest,
    HttpClient,
    RemoteFile,
    UrllibClient,
    fetch_file,
    intact,
    list_objects,
    read_manifest,
    write_manifest,
)
from physverify.datasets._layout import load_annotations, relative_to_data, resolve_data_path
from physverify.schema.clip import ClipManifest, FrameSampling, ObjectQuery, Split, TimeSource

if TYPE_CHECKING:
    from physverify.config.models import Settings

SECTION = "physics_iq"
TESTING = "testing-videos"
MASKS = "video-masks"
LOCAL = "local"
FULL = "full"
REVISION = "unversioned bucket, md5 per object"
NAME_PATTERN = re.compile(
    r"^(\d{4})_(testing-videos|video-masks)_(\d+)FPS_perspective-(\w+)_take-(\d)_trimmed-(.+)\.mp4$"
)

log = get_logger("datasets.physics_iq")
GreyFrames = NDArray[np.uint8]


class PhysicsIQSettings(FrozenModel):
    bucket_url: str
    listing_url: str
    license: str
    fps_file: int = Field(gt=0)
    view: str
    takes: tuple[int, ...] = Field(min_length=1)
    max_clips: int = Field(ge=0)
    max_frames: int = Field(ge=0)
    local: tuple[str, ...]
    full_dynamic: tuple[str, ...]
    full_static: tuple[str, ...]
    retries: int = Field(ge=0)


@dataclass(frozen=True)
class ObjectName:
    index: int
    kind: str
    fps: int
    view: str
    take: int
    scenario: str


def parse_name(name: str) -> ObjectName | None:
    """Fields of a bucket object name (the directory part is ignored)."""
    match = NAME_PATTERN.match(name.rsplit("/", 1)[-1])
    if match is None:
        return None
    index, kind, fps, view, take, scenario = match.groups()
    return ObjectName(int(index), kind, int(fps), view, int(take), scenario)


def prefixes(fps: int) -> dict[str, str]:
    return {TESTING: f"split-videos/testing/{fps}FPS/", MASKS: f"video-masks/real/{fps}FPS/"}


def clip_id_of(scenario: str, view: str, take: int) -> str:
    return f"{scenario}__{view}_take{take}"


def subset_scenarios(s: PhysicsIQSettings, subset: str) -> tuple[str, ...]:
    if subset == LOCAL:
        return s.local
    if subset == FULL:
        return (*s.full_dynamic, *s.full_static)
    raise UsageError(f"unknown physics_iq subset {subset!r}; known: {LOCAL}, {FULL}")


@dataclass(frozen=True)
class ClipFiles:
    """The testing video and the mask of one clip, as paths relative to the dataset root."""

    clip_id: str
    scenario: str
    testing: str
    mask: str


def expected_clips(s: PhysicsIQSettings, subset: str) -> tuple[tuple[str, int], ...]:
    """(scenario, take) of the subset in configuration order, capped at ``max_clips``."""
    pairs = tuple((scenario, take) for scenario in subset_scenarios(s, subset) for take in s.takes)
    return pairs[: s.max_clips or None]


def expected_ids(s: PhysicsIQSettings, subset: str, views: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        clip_id_of(scenario, view, take)
        for scenario, take in expected_clips(s, subset)
        for view in views
    )


def select_files(
    names: Iterable[str], s: PhysicsIQSettings, subset: str, views: Sequence[str] | None = None
) -> tuple[ClipFiles, ...]:
    """Pair testing video and mask per expected clip and view (default: the configured view);
    the lowest index wins a duplicate."""
    views = (s.view,) if views is None else tuple(views)
    found: dict[tuple[str, str, str, int], str] = {}
    for name in sorted(names):
        parsed = parse_name(name)
        if parsed is None or parsed.fps != s.fps_file or parsed.view not in views:
            continue
        found.setdefault((parsed.kind, parsed.view, parsed.scenario, parsed.take), name)
    selected: list[ClipFiles] = []
    missing: list[str] = []
    for scenario, take in expected_clips(s, subset):
        for view in views:
            clip_id = clip_id_of(scenario, view, take)
            key = (view, scenario, take)
            testing, mask = (found.get((kind, *key)) for kind in (TESTING, MASKS))
            if testing is None or mask is None:
                missing.append(clip_id)
                continue
            selected.append(ClipFiles(clip_id, scenario, testing, mask))
    if missing:
        raise MissingInputError(f"physics_iq: no testing video and mask for {missing}")
    return tuple(selected)


class PhysicsIQDataset:
    name: ClassVar[str] = "physics_iq"

    def __init__(self, settings: Settings, client: HttpClient | None = None) -> None:
        self._s = settings.section(SECTION, PhysicsIQSettings)
        self._views = self.views_of(settings)
        paths = settings.paths
        self._root = paths.raw_dir / self.name
        self._data_dir = paths.data_dir
        self._annotations = paths.annotations_dir / f"{SECTION}.json"
        self._max_frame_bytes = settings.runtime.max_frame_bytes
        self._client = client
        self._fetch_hint = f"physverify data fetch {self.name}"

    def views_of(self, settings: Settings) -> tuple[str, ...]:
        return (self._s.view,)

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        names = [str(p.relative_to(self._root)) for p in self._local_files()]
        try:
            files = select_files(names, self._s, subset or LOCAL, self._views)
        except MissingInputError as exc:
            raise MissingInputError(f"{exc} under {self._root}; run `{self._fetch_hint}`") from exc
        queries = load_annotations(self._annotations)
        return [self._manifest(f, queries) for f in files]

    def video(self, clip: ClipManifest) -> VideoSource:
        return Mp4VideoSource(self._path(clip.video_path), self._max_frame_bytes)

    def ground_truth(self, clip: ClipManifest) -> GroundTruthStore | None:
        return None

    def motion_masks(self, clip: ClipManifest, sampling: FrameSampling) -> GreyFrames:
        """Grey levels (T, h, w) of the clip's pixel-change mask on the sampled frames and size."""
        mask_path = self._mask_path(clip)
        source = Mp4VideoSource(mask_path, self._max_frame_bytes)
        indices = sampling.frame_indices
        frames = source.frames(
            int(indices[0]), int(indices[-1]) + 1, sampling.step, sampling.max_side
        )
        if frames.shape[1:3] != (sampling.height, sampling.width):
            raise MissingInputError(
                f"{mask_path.name}: mask size {frames.shape[2]}x{frames.shape[1]} differs "
                f"from the sampled video size {sampling.width}x{sampling.height}"
            )
        grey: GreyFrames = frames.max(axis=-1)
        return grey

    def fetch(self, subset: str | None, dest: Path) -> None:
        s = self._s
        subset = subset or LOCAL
        manifest_path = dest / MANIFEST_FILE
        wanted = expected_ids(s, subset, self._views)
        if _up_to_date(read_manifest(manifest_path), wanted, dest):
            log.info("%s: %d clips already fetched", self.name, len(wanted))
            return
        client = self._client or UrllibClient()
        listing = {
            str(item["name"]): item
            for prefix in prefixes(s.fps_file).values()
            for item in list_objects(client, s.listing_url, prefix, s.retries)
        }
        files = select_files(listing, s, subset, self._views)
        remotes = [self._remote(listing[n]) for f in files for n in (f.testing, f.mask)]
        for remote in remotes:
            fetch_file(client, remote, dest, s.retries)
        write_manifest(
            manifest_path,
            FetchManifest(
                dataset=self.name,
                source=s.bucket_url,
                revision=REVISION,
                license=s.license,
                subset=subset,
                fetched_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                files=tuple(
                    FetchedFile(relative=r.relative, size=r.size, md5_b64=r.md5_b64)
                    for r in remotes
                ),
                clips=tuple(f.clip_id for f in files),
            ),
        )

    def _remote(self, item: Mapping[str, Any]) -> RemoteFile:
        name = str(item["name"])
        return RemoteFile(
            url=f"{self._s.bucket_url}/{name}",
            relative=name,
            size=int(item["size"]),
            md5_b64=str(item["md5Hash"]),
        )

    def _local_files(self) -> list[Path]:
        folders = (self._root / p for p in prefixes(self._s.fps_file).values())
        return [path for folder in folders if folder.is_dir() for path in folder.glob("*.mp4")]

    def _path(self, relative: str) -> Path:
        path = resolve_data_path(relative, self._data_dir)
        if not path.is_file():
            raise MissingInputError(f"{path} is missing; run `{self._fetch_hint}`")
        return path

    def _mask_path(self, clip: ClipManifest) -> Path:
        testing = self._path(clip.video_path)
        mask_name = testing.name.replace(f"_{TESTING}_", f"_{MASKS}_", 1)
        return self._path(str(self._root / prefixes(self._s.fps_file)[MASKS] / mask_name))

    def _manifest(
        self, files: ClipFiles, queries: Mapping[str, tuple[ObjectQuery, ...]]
    ) -> ClipManifest:
        path = self._root / files.testing
        source = Mp4VideoSource(path, self._max_frame_bytes)
        width, height = source.native_size
        cap = self._s.max_frames or source.n_frames
        return ClipManifest(
            dataset=self.name,
            clip_id=files.clip_id,
            split=Split.EVAL,
            video_path=relative_to_data(path, self._data_dir),
            n_frames=min(source.n_frames, cap),
            width=width,
            height=height,
            container_fps=float(self._s.fps_file),
            seconds_per_frame=1.0 / self._s.fps_file,
            time_source=TimeSource.CONTAINER_FPS,
            object_queries=queries.get(files.clip_id, queries.get(files.scenario, ())),
            scene_id=files.scenario,
            license=self._s.license,
        )

    def is_static(self, clip: ClipManifest) -> bool:
        return clip.scene_id in self._s.full_static


def _up_to_date(manifest: FetchManifest | None, wanted: Sequence[str], dest: Path) -> bool:
    """True when a previous fetch covers ``wanted`` and its files are intact (no network)."""
    if manifest is None or not set(wanted) <= set(manifest.clips):
        return False
    return all(intact(dest / f.relative, f) for f in manifest.files)


PLUGINS = (PhysicsIQDataset,)
