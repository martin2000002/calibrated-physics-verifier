"""Resumable, verified downloads shared by the real datasets.

Bytes land in ``<file>.part`` and are renamed into place only after their size and checksum
match, so an interrupted fetch never leaves a file that looks complete.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import http.client
import json
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, BinaryIO, Protocol

from pydantic import ValidationError

from physverify.core.atomic import write_text_atomic
from physverify.core.digests import CHUNK_BYTES, file_sha256
from physverify.core.errors import PhysverifyError
from physverify.core.log import get_logger
from physverify.core.models import FrozenModel

PART_SUFFIX = ".part"
MANIFEST_FILE = "manifest.json"
BYTES_PER_MB = 1_000_000
HTTP_OK = 200
HTTP_PARTIAL = 206
TIMEOUT_S = 60.0
BACKOFF_S = 2.0
LISTING_FIELDS = "items(name,size,md5Hash),nextPageToken"

log = get_logger("datasets.download")
Sleep = Callable[[float], None]


class ChecksumError(PhysverifyError):
    """A downloaded or extracted file does not match its published size or checksum."""


class IncompleteDownloadError(OSError):
    """The server closed the body before the published size was reached (retried)."""


TRANSIENT = (OSError, http.client.HTTPException)


class HttpClient(Protocol):
    def get_json(self, url: str, params: Mapping[str, str]) -> Mapping[str, Any]: ...

    def open_range(self, url: str, start: int) -> tuple[int, BinaryIO]:
        """``(status, body)`` of a GET asking for ``bytes=start-``."""
        ...


def ascii_url(url: str) -> str:
    """Percent-encodes the non-ASCII characters of ``url`` (dataset file names carry ``’``);
    reserved characters and existing escapes are kept."""
    return urllib.parse.quote(url, safe=":/?#[]@!$&'()*+,;=%~")


class UrllibClient:
    """Standard-library client with a timeout; retries are the caller's (``retrying``)."""

    def __init__(self, timeout_s: float = TIMEOUT_S) -> None:
        self._timeout_s = timeout_s

    def get_json(self, url: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        query = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f"{url}?{query}", timeout=self._timeout_s) as response:
            data: Mapping[str, Any] = json.load(response)
        return data

    def open_range(self, url: str, start: int) -> tuple[int, BinaryIO]:
        headers = {"Range": f"bytes={start}-"} if start > 0 else {}
        request = urllib.request.Request(ascii_url(url), headers=headers)
        response = urllib.request.urlopen(request, timeout=self._timeout_s)
        return int(response.status), response


@dataclass(frozen=True)
class RemoteFile:
    url: str
    relative: str
    size: int
    md5_b64: str


@dataclass(frozen=True)
class FetchOutcome:
    path: Path
    transferred: int
    cached: bool


class FetchedFile(FrozenModel):
    relative: str
    size: int
    md5_b64: str | None = None
    sha256: str | None = None


class FetchManifest(FrozenModel):
    dataset: str
    source: str
    revision: str
    license: str
    subset: str
    fetched_utc: str
    files: tuple[FetchedFile, ...]
    clips: tuple[str, ...]


def retrying[T](action: Callable[[], T], retries: int, what: str, sleep: Sleep = time.sleep) -> T:
    """Run ``action``, retrying transient network errors with a doubling backoff."""
    attempt = 0
    while True:
        try:
            return action()
        except TRANSIENT as exc:
            if attempt >= retries:
                raise PhysverifyError(f"{what} failed after {attempt + 1} attempts: {exc}") from exc
            log.warning("%s: %s; retrying", what, exc)
            sleep(BACKOFF_S * 2**attempt)
            attempt += 1


def list_objects(
    client: HttpClient, listing_url: str, prefix: str, retries: int
) -> list[Mapping[str, Any]]:
    """Every object under ``prefix`` of a GCS bucket listing, following ``nextPageToken``."""
    items: list[Mapping[str, Any]] = []
    params = {"prefix": prefix, "fields": LISTING_FIELDS}
    while True:
        request = functools.partial(client.get_json, listing_url, dict(params))
        page = retrying(request, retries, f"listing {prefix}")
        items += page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            return items
        params["pageToken"] = str(token)


def md5_b64(path: Path) -> str:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "md5").digest()
    return base64.b64encode(digest).decode("ascii")


def intact(path: Path, record: FetchedFile) -> bool:
    """Size plus the digest the manifest recorded: md5 from a GCS listing, else sha256."""
    if not path.is_file() or path.stat().st_size != record.size:
        return False
    if record.sha256 is not None:
        return file_sha256(path) == record.sha256
    return record.md5_b64 is not None and md5_b64(path) == record.md5_b64


def part_path(path: Path) -> Path:
    return path.with_name(path.name + PART_SUFFIX)


def matches(path: Path, remote: RemoteFile) -> bool:
    return path.is_file() and path.stat().st_size == remote.size and md5_b64(path) == remote.md5_b64


def _append_range(client: HttpClient, remote: RemoteFile, part: Path) -> int:
    """Continue ``part`` from its current size: 206 appends, 200 restarts from zero."""
    start = part.stat().st_size if part.is_file() else 0
    if start > remote.size:
        part.unlink()
        start = 0
    if start == remote.size:
        return 0
    status, body = client.open_range(remote.url, start)
    with body:
        if status not in (HTTP_OK, HTTP_PARTIAL):
            raise PhysverifyError(f"{remote.url}: unexpected HTTP status {status}")
        with part.open("ab" if status == HTTP_PARTIAL else "wb") as out:
            return copy_stream(body, out)


def fetch_file(
    client: HttpClient, remote: RemoteFile, dest: Path, retries: int, sleep: Sleep = time.sleep
) -> FetchOutcome:
    """Download ``remote`` into ``dest/remote.relative``, resuming any ``.part`` left behind."""
    target = dest / remote.relative
    if matches(target, remote):
        return FetchOutcome(target, 0, cached=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(target)
    transferred = [0]

    def attempt() -> None:
        transferred[0] += _append_range(client, remote, part)
        if part.stat().st_size < remote.size:
            raise IncompleteDownloadError(f"{remote.relative}: body ended early")

    retrying(attempt, retries, f"download of {remote.relative}", sleep)
    if not matches(part, remote):
        part.unlink()
        raise ChecksumError(f"{remote.relative}: size or md5 differs from the listing")
    part.replace(target)
    log.info("%s: %d bytes", remote.relative, transferred[0])
    return FetchOutcome(target, transferred[0], cached=False)


def copy_stream(source: IO[bytes], out: IO[bytes]) -> int:
    written = 0
    for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
        out.write(chunk)
        written += len(chunk)
    return written


def save_atomic(source: IO[bytes], target: Path) -> int:
    """Stream ``source`` into ``target`` through a ``.part`` that is removed on any failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(target)
    try:
        with part.open("wb") as out:
            written = copy_stream(source, out)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(target)
    return written


def fetch_url(url: str, target: Path, timeout_s: float = TIMEOUT_S) -> int:
    """GET ``url`` into ``target`` through a ``.part`` (no resume, no checksum)."""
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return save_atomic(response, target)


def remote_size(url: str, timeout_s: float = TIMEOUT_S) -> int:
    """``Content-Length`` of a HEAD request, 0 when the server does not send it."""
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return int(response.headers.get("Content-Length", "0"))


def remote_zip(url: str) -> zipfile.ZipFile:
    """A zip read member by member through HTTP ranges (the archive is never downloaded whole)."""
    from remotezip import RemoteZip  # type: ignore[import-untyped]

    archive: zipfile.ZipFile = RemoteZip(url)
    return archive


def extract_member(archive: zipfile.ZipFile, member: str, target: Path) -> int:
    """Extract one member through a ``.part``; zipfile checks the CRC-32 at the end of the read."""
    try:
        with archive.open(member) as body:
            return save_atomic(body, target)
    except zipfile.BadZipFile as exc:
        raise ChecksumError(f"{member}: {exc}") from exc


def write_manifest(path: Path, manifest: FetchManifest) -> None:
    write_text_atomic(path, manifest.model_dump_json(indent=2) + "\n")


def read_manifest(path: Path) -> FetchManifest | None:
    """The manifest of a previous fetch, or None when there is none or it is unreadable."""
    if not path.is_file():
        return None
    try:
        return FetchManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        return None
