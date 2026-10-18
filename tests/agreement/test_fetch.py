"""``VideophyDataset.fetch`` against a fake hub and an in-memory zip (no network)."""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import pytest

from physverify.datasets._download import MANIFEST_FILE, ChecksumError, ascii_url, read_manifest
from physverify.datasets.videophy import VideophyDataset

TABLE = "train.csv"
PREFIX = "train_videos/"
MOVIE = b"\x00\x00\x00\x18ftypmp42 not really a movie"


class Hub:
    """Serves fixed bodies by URL and records every request."""

    def __init__(self, bodies: Mapping[str, bytes]) -> None:
        self.bodies = dict(bodies)
        self.requests: list[str] = []

    def get_json(self, url: str, params: Mapping[str, str]) -> Mapping[str, object]:
        raise AssertionError("no JSON requests expected")

    def open_range(self, url: str, start: int) -> tuple[int, BinaryIO]:
        self.requests.append(url)
        if url not in self.bodies:
            return 404, io.BytesIO(b"")
        return 200, io.BytesIO(self.bodies[url][start:])


class Opener:
    def __init__(self, members: Mapping[str, bytes]) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as out:
            for name, data in members.items():
                out.writestr(name, data)
        self.data = buffer.getvalue()
        self.opened = 0

    def __call__(self, url: str) -> zipfile.ZipFile:
        self.opened += 1
        return zipfile.ZipFile(io.BytesIO(self.data))


@pytest.fixture
def table(world) -> bytes:
    return (world.settings.paths.raw_dir / VideophyDataset.name / "labels" / TABLE).read_bytes()


def _setup(world, make_settings, table: bytes, pinned: str | None = None):
    sha = pinned or hashlib.sha256(table).hexdigest()
    settings = make_settings(f'videophy.parts.v1_dev.labels.sha256="{sha}"')
    part = settings.sections["videophy"]["parts"]["v1_dev"]
    hub_url = settings.sections["videophy"]["hub_url"]
    labels = part["labels"]
    table_url = f"{hub_url}/{labels['repo']}/resolve/{labels['revision']}/{TABLE}"
    hub = Hub({table_url: table, world.url("v1d/g3"): MOVIE})
    opener = Opener({f"{PREFIX}v1d/g1.mp4": MOVIE, f"{PREFIX}v1d/g2.mp4": b""})
    return VideophyDataset(settings, client=hub, opener=opener), hub, opener


def test_fetch_takes_members_from_the_zip_and_the_rest_from_their_url(
    world, make_settings, table: bytes, tmp_path: Path
) -> None:
    dataset, hub, opener = _setup(world, make_settings, table)
    dest = tmp_path / "dest"
    dataset.fetch("v1_dev", dest)
    videos = sorted(p.name for p in (dest / "videos" / "v1").iterdir())
    assert videos == sorted(f"{world.clip(p)}.mp4" for p in ("v1d/g1", "v1d/g3"))
    manifest = read_manifest(dest / MANIFEST_FILE)
    assert manifest is not None
    assert set(manifest.clips) == {world.clip("v1d/g1"), world.clip("v1d/g3")}
    assert opener.opened == 1 and world.url("v1d/g3") in hub.requests


def test_a_second_fetch_transfers_nothing(
    world, make_settings, table: bytes, tmp_path: Path
) -> None:
    dataset, hub, opener = _setup(world, make_settings, table)
    dest = tmp_path / "dest"
    dataset.fetch("v1_dev", dest)
    first = len(hub.requests)
    hub.bodies.pop(world.url("v1d/g3"))
    dataset.fetch("v1_dev", dest)
    assert opener.opened == 2  # g2 is empty upstream, so it is asked for again
    assert len(hub.requests) == first


def test_a_table_with_another_sha256_is_refused(
    world, make_settings, table: bytes, tmp_path: Path
) -> None:
    dataset, _, _ = _setup(world, make_settings, table, pinned="0" * 64)
    with pytest.raises(ChecksumError, match="pinned"):
        dataset.fetch("v1_dev", tmp_path / "dest")
    assert not (tmp_path / "dest" / "labels" / TABLE).exists()


def test_clips_list_only_the_videos_on_disk(world) -> None:
    assert world.dataset.clips("h2_dev") == []


def test_non_ascii_urls_are_percent_encoded_and_escapes_kept() -> None:
    url = "https://host/a/It’s%20here?x=1&y=2"
    assert ascii_url(url) == "https://host/a/It%E2%80%99s%20here?x=1&y=2"
    assert ascii_url("https://host/plain/file.mp4") == "https://host/plain/file.mp4"
