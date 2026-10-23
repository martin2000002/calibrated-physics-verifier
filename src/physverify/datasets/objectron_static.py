"""``objectron_static``: Objectron test videos of still objects, a sealed real control of the
freeze (plan-congelado section 3.2, J14).

Selection rule (Amendment 7): per class, the official ``test`` videos minus every video A9 looked
at (its seeded test draw and the end-to-end candidates), ordered by ``sha256(f"{seed}:{id}")``;
the first ``per_class`` whose MOV is at most ``max_video_mb`` are taken. A failed download or an
undecodable video gives its place to the next of the same order, never to another criterion. The
MOV is downloaded whole (its ``moov`` atom is not reliably first) and kept with its sha256; the
first ``seconds`` are decoded upright (the container rotation applied) into a frame directory.
The camera moves around the object: the control is still objects, not a still camera.
"""

from __future__ import annotations

import datetime as dt
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from physverify.core.atomic import write_text_atomic
from physverify.core.digests import file_sha256, text_sha256
from physverify.core.download import (
    BYTES_PER_MB,
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
from physverify.core.errors import MissingInputError, PhysverifyError, UsageError
from physverify.core.log import get_logger
from physverify.core.models import FrozenModel
from physverify.core.video import FrameDirVideoSource
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.datasets._layout import relative_to_data, resolve_data_path
from physverify.schema.clip import ClipManifest, ObjectQuery, Split, TimeSource
from physverify.sizing import draws, e2e, objectron
from physverify.sizing.settings import SizingSettings, sizing_settings

if TYPE_CHECKING:
    from physverify.config.models import Settings

SECTION = "objectron_static"
SELECTION_FILE = "selection.json"
FRAMES_DIR = "frames"
VIDEO_FILE = "video.MOV"
LOCAL, FULL = "local", "full"
SUBSETS = (LOCAL, FULL)
TAKEN, TOO_LARGE, FAILED = "taken", "too_large", "failed"
REVISION = "unversioned bucket, md5 per object; official {split} index"

log = get_logger("datasets.objectron_static")


class ObjectronStaticSettings(FrozenModel):
    per_class: int = Field(ge=1)
    seed: int
    seconds: float = Field(gt=0)
    max_video_mb: float = Field(gt=0)
    max_side: int = Field(ge=64)
    retries: int = Field(ge=0)
    split: Literal["train", "test"]


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


@dataclass(frozen=True)
class Decoded:
    fps: float
    n_frames: int
    width: int
    height: int


Decoder = Callable[[Path, Path, float, int], Decoded]
"""``(video, frame_dir, seconds, max_side)``: the decoded frames written under ``frame_dir``."""


def order_key(seed: int, video_id: str) -> str:
    return text_sha256(f"{seed}:{video_id}")


def ordered(
    refs: Iterable[objectron.VideoRef], excluded: frozenset[str], seed: int
) -> list[objectron.VideoRef]:
    """The candidates of one class in the preregistered order."""
    kept = (r for r in refs if r.video_id not in excluded)
    return sorted(kept, key=lambda r: order_key(seed, r.video_id))


def decode_upright(video: Path, target: Path, seconds: float, max_side: int) -> Decoded:
    """The first ``seconds`` at the native rate, rotation applied (``e2e.decode_frames``)."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if fps <= 0:
        return Decoded(0.0, 0, 0, 0)
    written = e2e.decode_frames(
        video, target, step=1, n_frames=round(seconds * fps), max_side=max_side
    )
    if written == 0:
        return Decoded(fps, 0, 0, 0)
    first = cv2.imread(str(target / "00000.jpg"))
    if first is None:
        return Decoded(fps, 0, 0, 0)
    return Decoded(fps, written, int(first.shape[1]), int(first.shape[0]))


@dataclass(frozen=True)
class _Outcome:
    candidate: Candidate
    clip: StaticClip | None = None
    file: FetchedFile | None = None


class ObjectronStaticDataset:
    name: ClassVar[str] = "objectron_static"

    def __init__(
        self,
        settings: Settings,
        client: HttpClient | None = None,
        reader: objectron.PrefixReader | None = None,
        decoder: Decoder | None = None,
    ) -> None:
        self._settings = settings
        self._s = settings.section(SECTION, ObjectronStaticSettings)
        self._sizing: SizingSettings = sizing_settings(settings)
        self._root = settings.paths.raw_dir / self.name
        self._data_dir = settings.paths.data_dir
        self._max_frame_bytes = settings.runtime.max_frame_bytes
        self._client = client
        self._reader = reader
        self._decoder = decoder or decode_upright
        self._fetch_hint = f"physverify data fetch {self.name}"

    # --- Reading ------------------------------------------------------------------------------

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        """Every selected clip: the control has one fixed selection, so ``local`` and ``full``
        (the subsets ``calibrate evaluate`` passes) are the same clips."""
        _check_subset(subset)
        chosen = read_selection(self._root / SELECTION_FILE)
        if chosen is None:
            raise MissingInputError(f"no selection under {self._root}; run `{self._fetch_hint}`")
        return [self._manifest(clip) for clip in chosen.clips]

    def video(self, clip: ClipManifest) -> VideoSource:
        path = resolve_data_path(clip.video_path, self._data_dir)
        if not path.is_dir():
            raise MissingInputError(f"{path} is missing; run `{self._fetch_hint}`")
        return FrameDirVideoSource(path, clip.real_fps, self._max_frame_bytes)

    def ground_truth(self, clip: ClipManifest) -> GroundTruthStore | None:
        return None

    def _manifest(self, clip: StaticClip) -> ClipManifest:
        frames = self._root / FRAMES_DIR / clip.clip_id
        return ClipManifest(
            dataset=self.name,
            clip_id=clip.clip_id,
            split=Split.EVAL,
            video_path=relative_to_data(frames, self._data_dir),
            n_frames=clip.n_frames,
            width=clip.width,
            height=clip.height,
            container_fps=clip.fps,
            seconds_per_frame=1.0 / clip.fps,
            time_source=TimeSource.CONTAINER_FPS,
            object_queries=(ObjectQuery(phrase=clip.label.replace("_", " "), expected_count=None),),
            scene_id=clip.video_id,
            license=self._sizing.license,
        )

    # --- Fetching -----------------------------------------------------------------------------

    def fetch(self, subset: str | None, dest: Path) -> None:
        subset = _check_subset(subset)
        previous = read_selection(dest / SELECTION_FILE)
        client = self._client or UrllibClient()
        if previous is not None:
            self._refuse_other_rule(previous)
            if _up_to_date(read_manifest(dest / MANIFEST_FILE), previous, dest):
                log.info("%s: %d clips already fetched", self.name, len(previous.clips))
            else:
                self._restore(client, dest, subset, previous)
            return
        reader = self._reader or objectron.UrllibPrefixReader()
        excluded = self._excluded(client, reader)
        index = draws.official_split(self._settings, self._sizing, self._s.split, reader)
        outcomes = [
            outcome
            for label in self._sizing.classes
            for outcome in self._select_class(client, dest, index[label], excluded)
        ]
        self._write(dest, subset, excluded, outcomes)

    def _restore(
        self, client: HttpClient, dest: Path, subset: str, previous: StaticSelection
    ) -> None:
        """Download again the clips of a selection already made; never a replacement (J14)."""
        refs = (objectron.VideoRef(c.video_id) for c in previous.clips)
        outcomes = [self._try(client, dest, ref) for ref in refs]
        lost = [o.candidate.video_id for o in outcomes if o.clip is None]
        if lost:
            raise MissingInputError(
                f"{self.name}: selected videos {lost} cannot be restored; the selection is "
                "fixed once made (J14)"
            )
        self._write_manifest(dest, subset, outcomes)

    def _refuse_other_rule(self, previous: StaticSelection) -> None:
        if previous.rule() != self._rule():
            raise UsageError(
                f"{self.name}: a selection with seed/split/per_class/max_video_mb/seconds "
                f"{previous.rule()} exists, this configuration asks for {self._rule()}; "
                "the selection is fixed once made (J14)"
            )

    def _rule(self) -> tuple[Any, ...]:
        s = self._s
        return (s.seed, s.split, s.per_class, s.max_video_mb, s.seconds)

    def _excluded(self, client: HttpClient, reader: objectron.PrefixReader) -> frozenset[str]:
        """The A9 draw of the split and every end-to-end candidate A9 examined."""
        tested = draws.selection(self._settings, self._sizing, self._s.split, reader)
        limit = self._sizing.e2e.max_video_mb * BYTES_PER_MB

        def fits(ref: objectron.VideoRef) -> bool:
            listed = self._listed(client, ref)
            return listed is not None and int(listed["size"]) <= limit

        examined = draws.e2e_draw(self._settings, self._sizing, self._sizing.e2e, reader, fits)
        return frozenset(r.video_id for r in tested) | examined.examined

    def _select_class(
        self,
        client: HttpClient,
        dest: Path,
        refs: Sequence[objectron.VideoRef],
        excluded: frozenset[str],
    ) -> list[_Outcome]:
        outcomes: list[_Outcome] = []
        taken = 0
        for ref in ordered(refs, excluded, self._s.seed):
            if taken == self._s.per_class:
                break
            outcome = self._try(client, dest, ref)
            outcomes.append(outcome)
            if outcome.clip is not None:
                taken += 1
        if taken < self._s.per_class and refs:
            log.warning("%s: %s has only %d clips", self.name, refs[0].label, taken)
        return outcomes

    def _try(self, client: HttpClient, dest: Path, ref: objectron.VideoRef) -> _Outcome:
        listed = self._listed(client, ref)
        if listed is None:
            return _Outcome(Candidate(video_id=ref.video_id, status=FAILED))
        remote = self._remote(listed)
        if remote.size > self._s.max_video_mb * BYTES_PER_MB:
            return _Outcome(Candidate(video_id=ref.video_id, status=TOO_LARGE, size=remote.size))
        failed = _Outcome(Candidate(video_id=ref.video_id, status=FAILED, size=remote.size))
        try:
            path = fetch_file(client, remote, dest, self._s.retries).path
        except (OSError, PhysverifyError) as exc:
            log.warning("%s: %s failed (%s); the next of the order is taken", self.name, ref, exc)
            return failed
        decoded = self._decode(path, dest / FRAMES_DIR / e2e.clip_id_of(ref))
        if decoded.n_frames == 0:
            log.warning("%s: %s decodes to no frame; the next is taken", self.name, ref.video_id)
            path.unlink()
            return failed
        return _Outcome(
            Candidate(video_id=ref.video_id, status=TAKEN, size=remote.size),
            StaticClip(
                clip_id=e2e.clip_id_of(ref),
                video_id=ref.video_id,
                label=ref.label,
                fps=decoded.fps,
                n_frames=decoded.n_frames,
                width=decoded.width,
                height=decoded.height,
            ),
            FetchedFile(
                relative=remote.relative,
                size=remote.size,
                md5_b64=remote.md5_b64,
                sha256=file_sha256(path),
            ),
        )

    def _decode(self, video: Path, target: Path) -> Decoded:
        """Into a ``.part`` directory renamed when whole, so a frame directory is never partial."""
        part = target.with_name(target.name + ".part")
        for stale in (part, target):
            if stale.exists():
                shutil.rmtree(stale)
        decoded = self._decoder(video, part, self._s.seconds, self._s.max_side)
        if decoded.n_frames == 0:
            shutil.rmtree(part, ignore_errors=True)
        else:
            part.replace(target)
        return decoded

    def _listed(self, client: HttpClient, ref: objectron.VideoRef) -> Mapping[str, Any] | None:
        name = f"{objectron.video_prefix(ref)}{VIDEO_FILE}"
        items = list_objects(client, self._sizing.listing_url, name, self._s.retries)
        return next((item for item in items if item.get("name") == name), None)

    def _remote(self, item: Mapping[str, Any]) -> RemoteFile:
        name = str(item["name"])
        return RemoteFile(
            url=f"{self._sizing.bucket_url}/{name}",
            relative=name,
            size=int(item["size"]),
            md5_b64=str(item["md5Hash"]),
        )

    def _write(
        self, dest: Path, subset: str, excluded: frozenset[str], outcomes: Sequence[_Outcome]
    ) -> None:
        s = self._s
        chosen = StaticSelection(
            seed=s.seed,
            split=s.split,
            per_class=s.per_class,
            max_video_mb=s.max_video_mb,
            seconds=s.seconds,
            excluded=tuple(sorted(excluded)),
            candidates=tuple(o.candidate for o in outcomes),
            clips=tuple(o.clip for o in outcomes if o.clip is not None),
        )
        self._write_manifest(dest, subset, outcomes)
        write_text_atomic(dest / SELECTION_FILE, chosen.model_dump_json(indent=1) + "\n")
        log.info("%s ids: %s", self.name, ", ".join(c.video_id for c in chosen.clips))

    def _write_manifest(self, dest: Path, subset: str, outcomes: Sequence[_Outcome]) -> None:
        write_manifest(
            dest / MANIFEST_FILE,
            FetchManifest(
                dataset=self.name,
                source=self._sizing.bucket_url,
                revision=REVISION.format(split=self._s.split),
                license=self._sizing.license,
                subset=subset,
                fetched_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                files=tuple(o.file for o in outcomes if o.file is not None),
                clips=tuple(o.clip.clip_id for o in outcomes if o.clip is not None),
            ),
        )


def _check_subset(subset: str | None) -> str:
    subset = subset or LOCAL
    if subset not in SUBSETS:
        raise UsageError(f"unknown objectron_static subset {subset!r}; known: {list(SUBSETS)}")
    return subset


def read_selection(path: Path) -> StaticSelection | None:
    if not path.is_file():
        return None
    return StaticSelection.model_validate_json(path.read_text(encoding="utf-8"))


def _up_to_date(manifest: FetchManifest | None, chosen: StaticSelection, dest: Path) -> bool:
    """True when every selected clip has its intact MOV and its frame directory (no network)."""
    if manifest is None or set(manifest.clips) != {c.clip_id for c in chosen.clips}:
        return False
    frames = (dest / FRAMES_DIR / c.clip_id for c in chosen.clips)
    return all(intact(dest / f.relative, f) for f in manifest.files) and all(
        f.is_dir() for f in frames
    )


PLUGINS = (ObjectronStaticDataset,)
