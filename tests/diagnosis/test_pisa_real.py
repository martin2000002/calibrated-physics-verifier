"""PISA real drops: prompt queries, seeded selection, verified member extraction."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path

import cv2
import numpy as np
import pytest

from physverify.config.models import Settings
from physverify.core.digests import file_sha256
from physverify.core.download import MANIFEST_FILE, ChecksumError, part_path, read_manifest
from physverify.core.errors import MissingInputError, UsageError
from physverify.datasets.pisa_real import (
    INDEX_FILE,
    PisaRealDataset,
    derive_queries,
    select_ids,
)
from physverify.schema.clip import ObjectQuery, TimeSource

SIZE = (32, 24)
N_FRAMES = 10
MAX_FRAMES = 6
CONTAINER_FPS = 30.0
PROMPTS = {
    "0001": "A tennis ball falls onto the table.",
    "0002": "The red cup is dropped.",
    "0003": "A white bottle tumbles and rolls.",
}

SettingsFactory = Callable[..., Settings]


class Opener:
    """An in-memory ``RemoteZip`` stand-in that counts how often the archive is opened."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.urls: list[str] = []

    def __call__(self, url: str) -> zipfile.ZipFile:
        self.urls.append(url)
        return zipfile.ZipFile(io.BytesIO(self.data))


def refuse(url: str) -> zipfile.ZipFile:
    raise AssertionError(f"the archive was opened: {url}")


def archive(movie: bytes, prompts: Mapping[str, str] = PROMPTS) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as out:
        out.writestr("real/README.txt", "not a clip")
        for clip_id, prompt in prompts.items():
            out.writestr(f"real/{clip_id}/movie.mp4", movie)
            out.writestr(f"real/{clip_id}/clip_info.json", json.dumps({"prompt": prompt}))
    return buffer.getvalue()


@pytest.fixture(scope="module")
def movie(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp("mp4") / "movie.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), CONTAINER_FPS, SIZE)
    for i in range(N_FRAMES):
        writer.write(np.full((SIZE[1], SIZE[0], 3), 25 * i, np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.fixture
def settings(make_settings: SettingsFactory) -> Settings:
    """Seeded selection over the fake archive: the repo config pins ids the fake does not have."""
    return make_settings(
        "pisa_real.max_clips=2",
        f"pisa_real.max_frames={MAX_FRAMES}",
        "pisa_real.local_ids=[]",
        fakes=False,
    )


def dest_of(settings: Settings) -> Path:
    return settings.paths.raw_dir / PisaRealDataset.name


@pytest.mark.parametrize(
    ("prompt", "phrase"),
    [
        ("A white bottle falls.", "white bottle"),
        ("The red ball is dropped onto the table", "red ball"),
        ("Ball rolls off, then falls!", "ball"),
        ("An orange tumbles", "orange"),
    ],
)
def test_the_query_is_the_subject_before_the_first_fall_verb(prompt: str, phrase: str) -> None:
    verbs = ("falls", "drops", "is dropped", "tumbles", "rolls")
    assert derive_queries(prompt, verbs) == (ObjectQuery(phrase, 1),)


def test_an_empty_prompt_gives_no_query() -> None:
    assert derive_queries("  ", ("falls",)) == ()


def test_selection_is_seeded_and_puts_size_priors_first() -> None:
    prompts = {f"{i:04d}": ("cup" if i % 3 == 0 else "thing") for i in range(12)}
    has_prior: Callable[[str], bool] = lambda prompt: prompt == "cup"  # noqa: E731
    first = select_ids(prompts, 5, 0, has_prior)
    assert first == select_ids(prompts, 5, 0, has_prior)
    assert len(first) == 5
    assert all(prompts[i] == "cup" for i in first[:4])
    assert prompts[first[4]] == "thing"
    assert select_ids(prompts, 0, 0, has_prior) != select_ids(prompts, 0, 1, has_prior)
    assert sorted(select_ids(prompts, 0, 0, has_prior)) == sorted(prompts)


def test_fetch_extracts_the_selected_clips_and_records_them(
    settings: Settings, movie: bytes
) -> None:
    opener = Opener(archive(movie))
    dataset = PisaRealDataset(settings, opener=opener)
    dest = dest_of(settings)
    dataset.fetch("local", dest)
    assert opener.urls == [dataset.archive_url]
    assert "/resolve/8f66e855fca40b7d9ab7ef947c51b52668adc762/pisabench/real.zip" in opener.urls[0]
    assert json.loads((dest / INDEX_FILE).read_text()) == PROMPTS
    manifest = read_manifest(dest / MANIFEST_FILE)
    assert manifest is not None
    expected = select_ids(PROMPTS, 2, 0, dataset.has_prior)
    assert manifest.clips == expected
    assert manifest.revision == "8f66e855fca40b7d9ab7ef947c51b52668adc762"
    assert [f.relative for f in manifest.files] == [
        f"{i}/{m}" for i in expected for m in ("movie.mp4", "clip_info.json")
    ]
    assert all(f.sha256 == file_sha256(dest / f.relative) for f in manifest.files)


def test_a_second_fetch_reuses_the_intact_files(settings: Settings, movie: bytes) -> None:
    dest = dest_of(settings)
    PisaRealDataset(settings, opener=Opener(archive(movie))).fetch("local", dest)
    first = read_manifest(dest / MANIFEST_FILE)
    PisaRealDataset(settings, opener=refuse).fetch("local", dest)
    second = read_manifest(dest / MANIFEST_FILE)
    assert first is not None and second is not None
    assert (second.files, second.clips) == (first.files, first.clips)


def test_clips_declare_the_capture_rate(settings: Settings, movie: bytes) -> None:
    dataset = PisaRealDataset(settings, opener=Opener(archive(movie)))
    dataset.fetch("local", dest_of(settings))
    clips = dataset.clips("local")
    assert [c.clip_id for c in clips] == list(select_ids(PROMPTS, 2, 0, dataset.has_prior))
    clip = clips[0]
    assert clip.time_source is TimeSource.DECLARED
    assert clip.seconds_per_frame == pytest.approx(1 / 120)
    assert clip.real_fps == pytest.approx(120.0)
    assert clip.container_fps == pytest.approx(CONTAINER_FPS)
    assert clip.n_frames == MAX_FRAMES
    assert clip.prompt == PROMPTS[clip.clip_id]
    assert clip.object_queries == derive_queries(
        clip.prompt or "", ("falls", "is dropped", "tumbles")
    )
    assert dataset.ground_truth(clip) is None
    assert dataset.video(clip).real_fps == pytest.approx(120.0)


def test_pinned_ids_and_annotations_win(make_settings: SettingsFactory, movie: bytes) -> None:
    settings = make_settings('pisa_real.local_ids=["0003"]', fakes=False)
    path = settings.paths.annotations_dir / "pisa_real.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"0003": [{"phrase": "plastic bottle", "expected_count": 1}]}))
    dataset = PisaRealDataset(settings, opener=Opener(archive(movie)))
    dataset.fetch("local", dest_of(settings))
    assert not (dest_of(settings) / INDEX_FILE).exists()
    (clip,) = dataset.clips("local")
    assert clip.clip_id == "0003"
    assert clip.object_queries == (ObjectQuery("plastic bottle", 1),)


def test_a_corrupt_member_is_never_promoted(settings: Settings, movie: bytes) -> None:
    payload = b"MOVIE-BYTES-0123456789"
    data = archive(payload).replace(payload, payload[:-1] + b"X")
    dataset = PisaRealDataset(settings, opener=Opener(data))
    dest = dest_of(settings)
    with pytest.raises(ChecksumError, match="CRC"):
        dataset.fetch("local", dest)
    movies = list(dest.glob("*/movie.mp4"))
    assert movies == []
    assert list(dest.glob("*/*.part")) == []
    assert not (dest / MANIFEST_FILE).exists()
    assert not part_path(dest / "0001" / "movie.mp4").exists()


def test_clips_before_fetch_point_to_the_fetch_command(settings: Settings) -> None:
    with pytest.raises(MissingInputError, match="download fetch pisa_real"):
        PisaRealDataset(settings, opener=refuse).clips("local")


def test_unknown_subset_is_a_usage_error(settings: Settings) -> None:
    with pytest.raises(UsageError, match="subset"):
        PisaRealDataset(settings, opener=refuse).clips("everything")
