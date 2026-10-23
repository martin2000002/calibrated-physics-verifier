"""VideoPhy and VideoPhy-2 generated videos with human physical-commonsense labels (ADR 0011 D1).

Label tables are pinned by revision and sha256; videos are read member by member from the
published zips through HTTP ranges (CRC-checked), or from their URL when the zip lacks them.
Test subsets come from the lexical object-object rule plus a recorded hand review; there is no
ground truth, and time comes from the container rate.
"""

from __future__ import annotations

import datetime as dt
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from physverify.core.digests import file_sha256
from physverify.core.download import (
    MANIFEST_FILE,
    ChecksumError,
    FetchedFile,
    FetchManifest,
    HttpClient,
    UrllibClient,
    extract_member,
    intact,
    read_manifest,
    remote_zip,
    retrying,
    save_atomic,
    write_manifest,
)
from physverify.core.errors import MissingInputError, PhysverifyError
from physverify.core.log import get_logger
from physverify.core.video import Mp4VideoSource
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.datasets._layout import load_annotations, relative_to_data, resolve_data_path
from physverify.datasets._queries import derive_queries
from physverify.datasets._videophy_review import load_review, reviewed_rows
from physverify.datasets._videophy_tables import (
    LABELS_DIR,
    SECTION,
    SMOKE,
    VIDEOS_DIR,
    HubFile,
    PartSettings,
    VideophyRow,
    VideophySettings,
    build_rows,
    in_scope,
    lexical_reason,
    member_map,
    read_table,
    reference_scores,
)
from physverify.schema.clip import ClipManifest, ObjectQuery, TimeSource

if TYPE_CHECKING:
    from physverify.config.models import Settings

FETCH_HINT = "physverify agreement fetch"
log = get_logger("datasets.videophy")
ArchiveOpener = Callable[[str], zipfile.ZipFile]


class VideophyDataset:
    name: ClassVar[str] = "videophy"

    def __init__(
        self,
        settings: Settings,
        client: HttpClient | None = None,
        opener: ArchiveOpener | None = None,
    ) -> None:
        self._s = settings.section(SECTION, VideophySettings)
        paths = settings.paths
        self._root = paths.raw_dir / self.name
        self._data_dir = paths.data_dir
        self._subset_path = paths.annotations_dir / self._s.subset_file
        self._queries_path = paths.annotations_dir / self._s.queries_file
        self._max_frame_bytes = settings.runtime.max_frame_bytes
        self._client = client or UrllibClient()
        self._opener = opener or remote_zip

    @property
    def settings(self) -> VideophySettings:
        return self._s

    @property
    def subset_path(self) -> Path:
        return self._subset_path

    @property
    def queries_path(self) -> Path:
        return self._queries_path

    def candidates(self, subset: str) -> dict[str, list[VideophyRow]]:
        """``{part: rows}`` kept by the scope column and the lexical rule, before the review."""
        return {name: self._candidates(name, self._root) for name in self._s.parts_of(subset)}

    def rows(self, subset: str | None = None) -> list[VideophyRow]:
        """Labelled rows of the subset: reviewed parts go through the recorded hand review."""
        subset = subset or SMOKE
        review = load_review(self._subset_path)
        rows: list[VideophyRow] = []
        for name, part_rows in self.candidates(subset).items():
            if name in self._s.reviewed_subsets:
                part_rows = reviewed_rows(review, name, part_rows, self._s.filter_version)
            rows.extend(part_rows)
        if subset == SMOKE:
            rows = [row for row in rows if row.clip_id in self._s.smoke_ids]
        return rows

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        """Manifests of the subset's videos that are on disk (the comparison counts the rest)."""
        queries = self.queries()
        present = [r for r in self.rows(subset) if self.video_file(r).is_file()]
        manifests = [self._manifest(row, queries) for row in present]
        return [m for m in manifests if m is not None]

    def queries(self) -> dict[str, tuple[ObjectQuery, ...]]:
        """``{caption: queries}`` of the queries file (D11: the unreviewed heuristic draft)."""
        return load_annotations(self._queries_path)

    def object_queries(
        self, row: VideophyRow, queries: Mapping[str, tuple[ObjectQuery, ...]]
    ) -> tuple[ObjectQuery, ...]:
        """The queries a clip is extracted with: the file's entry, else the caption heuristic;
        empty means the clip has no object to detect."""
        return queries.get(row.caption) or derive_queries(row.caption, self._s.filter.rigid_event)

    def video(self, clip: ClipManifest) -> VideoSource:
        path = resolve_data_path(clip.video_path, self._data_dir)
        return Mp4VideoSource(path, self._max_frame_bytes)

    def ground_truth(self, clip: ClipManifest) -> GroundTruthStore | None:
        return None

    def video_file(self, row: VideophyRow, root: Path | None = None) -> Path:
        return (root or self._root) / VIDEOS_DIR / row.version / f"{row.clip_id}.mp4"

    def fetch(self, subset: str | None, dest: Path) -> None:
        subset = subset or SMOKE
        previous = read_manifest(dest / MANIFEST_FILE)
        known = {} if previous is None else {f.relative: f for f in previous.files}
        names = self._s.parts_of(subset)
        tables = [self._fetch_table(f, dest, known) for f in self._hub_files(names)]
        wanted = {n: self._candidates(n, dest) for n in names}
        if subset == SMOKE:
            wanted = {
                n: [r for r in rows if r.clip_id in self._s.smoke_ids] for n, rows in wanted.items()
            }
        videos: list[FetchedFile] = []
        for name, rows in wanted.items():
            videos.extend(self._fetch_part(self._s.parts[name], rows, dest, known))
        self._write_manifest(dest, subset, previous, [*tables, *videos])

    def _hub_files(self, names: Sequence[str]) -> list[HubFile]:
        files: list[HubFile] = []
        for name in names:
            part = self._s.parts[name]
            files.extend(f for f in (part.labels, part.reference) if f is not None)
        return list({f.url(self._s.hub_url): f for f in files}.values())

    def _candidates(self, name: str, root: Path) -> list[VideophyRow]:
        part = self._s.parts[name]
        records = [r for r in read_table(self._table(part.labels, root)) if in_scope(r, part)]
        kept = [r for r in records if lexical_reason(r["caption"], self._s.filter) is None]
        reference: dict[str, float] = {}
        if part.reference is not None:
            reference = reference_scores(
                read_table(self._table(part.reference, root)), part.reference_column
            )
        return build_rows(name, part, kept, reference)

    def _table(self, hub: HubFile, root: Path) -> Path:
        path = root / LABELS_DIR / hub.file
        if not path.is_file():
            raise MissingInputError(f"{path} is missing; run `{FETCH_HINT}`")
        return path

    def _fetch_table(
        self, hub: HubFile, dest: Path, known: Mapping[str, FetchedFile]
    ) -> FetchedFile:
        relative = f"{LABELS_DIR}/{hub.file}"
        target = dest / relative
        record = known.get(relative)
        if record is not None and intact(target, record) and hub.sha256 in (None, record.sha256):
            return record
        size = self._download(hub.url(self._s.hub_url), target)
        digest = file_sha256(target)
        if hub.sha256 is not None and digest != hub.sha256:
            target.unlink()
            raise ChecksumError(f"{relative}: sha256 {digest} differs from the pinned {hub.sha256}")
        return FetchedFile(relative=relative, size=size, sha256=digest)

    def _download(self, url: str, target: Path) -> int:
        def attempt() -> int:
            status, body = self._client.open_range(url, 0)
            with body:
                if status != 200:
                    raise PhysverifyError(f"{url}: unexpected HTTP status {status}")
                return save_atomic(body, target)

        return retrying(attempt, self._s.retries, f"download of {url}")

    def _fetch_part(
        self,
        part: PartSettings,
        rows: Sequence[VideophyRow],
        dest: Path,
        known: Mapping[str, FetchedFile],
    ) -> list[FetchedFile]:
        """Every video of ``rows`` not already intact; a video that fails is logged and skipped."""
        missing = [r for r in rows if not self._cached(r, dest, known)]
        absent = {r.clip_id for r in missing}
        records = [known[self._relative(r, dest)] for r in rows if r.clip_id not in absent]
        if not missing:
            return records
        url = part.videos.url(self._s.hub_url)
        archive = retrying(lambda: self._opener(url), self._s.retries, f"open {url}")
        with archive:
            members = member_map(
                [r.video_url for r in missing], archive.namelist(), part.videos.prefix
            )
            for row in missing:
                fetched = self._fetch_video(archive, members.get(row.video_url), row, dest)
                records.extend([fetched] if fetched is not None else [])
        failed = len(rows) - len(records)
        log.info("%s: %d/%d videos (%d failed)", part.videos.file, len(records), len(rows), failed)
        return records

    def _relative(self, row: VideophyRow, dest: Path) -> str:
        return str(self.video_file(row, dest).relative_to(dest))

    def _cached(self, row: VideophyRow, dest: Path, known: Mapping[str, FetchedFile]) -> bool:
        record = known.get(self._relative(row, dest))
        return record is not None and intact(self.video_file(row, dest), record)

    def _fetch_video(
        self, archive: zipfile.ZipFile, member: str | None, row: VideophyRow, dest: Path
    ) -> FetchedFile | None:
        target = self.video_file(row, dest)
        try:
            if member is not None:
                retrying(lambda: extract_member(archive, member, target), self._s.retries, member)
            else:
                self._download(row.video_url, target)
        except PhysverifyError as exc:
            log.warning("%s (%s): %s", row.clip_id, row.video_url, exc)
            return None
        if target.stat().st_size == 0:
            log.warning("%s: empty video, discarded", row.clip_id)
            target.unlink()
            return None
        return FetchedFile(
            relative=self._relative(row, dest),
            size=target.stat().st_size,
            sha256=file_sha256(target),
        )

    def _write_manifest(
        self, dest: Path, subset: str, previous: FetchManifest | None, files: Sequence[FetchedFile]
    ) -> None:
        merged = {} if previous is None else {f.relative: f for f in previous.files}
        merged.update({f.relative: f for f in files})
        ids = {Path(r).stem for r in merged if r.startswith(f"{VIDEOS_DIR}/")}
        write_manifest(
            dest / MANIFEST_FILE,
            FetchManifest(
                dataset=self.name,
                source=self._s.hub_url,
                revision=self._revisions(),
                license=self._s.license,
                subset=subset,
                fetched_utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                files=tuple(sorted(merged.values(), key=lambda f: f.relative)),
                clips=tuple(sorted(ids)),
            ),
        )

    def _revisions(self) -> str:
        pinned = {
            f"{hub.repo}@{hub.revision}"
            for part in self._s.parts.values()
            for hub in (part.labels, part.reference, part.videos)
            if hub is not None
        }
        return "; ".join(sorted(pinned))

    def _manifest(
        self, row: VideophyRow, queries: Mapping[str, tuple[ObjectQuery, ...]]
    ) -> ClipManifest | None:
        """None (logged) for a video OpenCV cannot read: no frames or no container rate."""
        path = self.video_file(row)
        source = Mp4VideoSource(path, self._max_frame_bytes)
        if source.n_frames <= 0 or source.real_fps <= 0:
            log.warning("%s: unreadable video %s, left out", row.clip_id, path)
            return None
        width, height = source.native_size
        cap = self._s.max_frames or source.n_frames
        return ClipManifest(
            dataset=self.name,
            clip_id=row.clip_id,
            split=row.split,
            video_path=relative_to_data(path, self._data_dir),
            n_frames=min(source.n_frames, cap),
            width=width,
            height=height,
            container_fps=source.real_fps,
            seconds_per_frame=1.0 / source.real_fps,
            time_source=TimeSource.CONTAINER_FPS,
            object_queries=self.object_queries(row, queries),
            prompt=row.caption,
            license=self._s.license,
        )


PLUGINS = (VideophyDataset,)
