"""Physics-IQ (ADR 0007 §4.2): bucket listing, resumable verified download, local clips."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np
import pytest

from physverify.config.models import Settings
from physverify.core.errors import MissingInputError, UsageError
from physverify.datasets._download import (
    MANIFEST_FILE,
    ChecksumError,
    RemoteFile,
    fetch_file,
    part_path,
    read_manifest,
)
from physverify.datasets._layout import load_annotations
from physverify.datasets.physics_iq import (
    MASKS,
    TESTING,
    PhysicsIQDataset,
    PhysicsIQSettings,
    list_objects,
    parse_name,
    select_files,
)
from physverify.schema.clip import FrameSampling, ObjectQuery, TimeSource

BUCKET = "https://bucket.test/physics-iq"
LISTING = "https://listing.test/o"
SIZE = (32, 24)
N_FRAMES = 12
MAX_FRAMES = 8
REPO_ANNOTATIONS = Path(__file__).resolve().parents[2] / "annotations"

SettingsFactory = Callable[..., Settings]


def md5_of(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode("ascii")


def object_name(
    scenario: str, kind: str = TESTING, *, index: int = 1, fps: int = 16, **kw: Any
) -> str:
    view, take = kw.get("view", "center"), kw.get("take", 1)
    folder = f"split-videos/testing/{fps}FPS" if kind == TESTING else f"video-masks/real/{fps}FPS"
    stem = f"{index:04d}_{kind}_{fps}FPS_perspective-{view}_take-{take}_trimmed-{scenario}"
    return f"{folder}/{stem}.mp4"


class FakeBucket:
    """The GCS JSON listing and object GETs, with pages and optional early-closed bodies."""

    def __init__(self, objects: Mapping[str, bytes], page_size: int = 100) -> None:
        self.objects = dict(objects)
        self.page_size = page_size
        self.md5 = {name: md5_of(data) for name, data in objects.items()}
        self.cut: dict[str, int] = {}
        self.listings: list[Mapping[str, str]] = []
        self.ranges: list[tuple[str, int]] = []

    def get_json(self, url: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        self.listings.append(dict(params))
        names = sorted(n for n in self.objects if n.startswith(params["prefix"]))
        start = int(params.get("pageToken", "0"))
        page = names[start : start + self.page_size]
        items = [
            {"name": n, "size": str(len(self.objects[n])), "md5Hash": self.md5[n]} for n in page
        ]
        more = start + self.page_size < len(names)
        return {"items": items, **({"nextPageToken": str(start + self.page_size)} if more else {})}

    def open_range(self, url: str, start: int) -> tuple[int, BinaryIO]:
        name = url.removeprefix(f"{BUCKET}/")
        self.ranges.append((name, start))
        body = self.objects[name][start:]
        if name in self.cut:
            body = body[: self.cut.pop(name)]
        return (206 if start else 200), io.BytesIO(body)

    @property
    def calls(self) -> int:
        return len(self.listings) + len(self.ranges)


def remote(name: str, data: bytes, md5: str | None = None) -> RemoteFile:
    return RemoteFile(
        url=f"{BUCKET}/{name}", relative=name, size=len(data), md5_b64=md5 or md5_of(data)
    )


def no_sleep(_: float) -> None:
    return None


@pytest.fixture
def settings(make_settings: SettingsFactory) -> Settings:
    return make_settings(
        f'physics_iq.bucket_url="{BUCKET}"',
        f'physics_iq.listing_url="{LISTING}"',
        f"physics_iq.max_frames={MAX_FRAMES}",
        fakes=False,
    )


@pytest.fixture
def section(settings: Settings) -> PhysicsIQSettings:
    return settings.section("physics_iq", PhysicsIQSettings)


@pytest.fixture(scope="module")
def mp4_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp("mp4") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 16.0, SIZE)
    for i in range(N_FRAMES):
        writer.write(np.full((SIZE[1], SIZE[0], 3), 20 * i, np.uint8))
    writer.release()
    return path.read_bytes()


def local_objects(section: PhysicsIQSettings, data: bytes) -> dict[str, bytes]:
    return {
        object_name(scenario, kind, index=i + 1): data
        for i, scenario in enumerate(section.local)
        for kind in (TESTING, MASKS)
    }


# --- Names and selection ----------------------------------------------------------------------


def test_parse_name_reads_every_field() -> None:
    parsed = parse_name(object_name("ball-and-block-fall", MASKS, index=7, fps=30, take=2))
    assert parsed is not None
    assert (parsed.index, parsed.kind, parsed.fps, parsed.view, parsed.take) == (
        7,
        MASKS,
        30,
        "center",
        2,
    )
    assert parsed.scenario == "ball-and-block-fall"
    assert parse_name("split-videos/testing/16FPS/readme.txt") is None


def test_local_subset_is_four_dynamic_clips_and_one_static(section: PhysicsIQSettings) -> None:
    names = list(local_objects(section, b""))
    names += [
        object_name("ball-and-block-fall", index=99),
        object_name("balls-collide", fps=30),
        object_name("balls-collide", view="left"),
        object_name("balls-collide", take=2),
        object_name("ball-hits-duck"),
    ]
    files = select_files(names, section, "local")
    assert [f.scenario for f in files] == list(section.local)
    assert files[0].clip_id == "ball-and-block-fall__center_take1"
    assert files[0].testing == object_name("ball-and-block-fall", index=1)
    assert files[0].mask == object_name("ball-and-block-fall", MASKS, index=1)
    static = [f.scenario for f in files if f.scenario in section.full_static]
    assert static == ["stable-blocks"]


def test_a_missing_mask_names_the_clip(section: PhysicsIQSettings) -> None:
    names = [n for n in local_objects(section, b"") if "balls-collide" not in n or TESTING in n]
    with pytest.raises(MissingInputError, match="balls-collide__center_take1"):
        select_files(names, section, "local")


def test_unknown_subset_is_a_usage_error(section: PhysicsIQSettings) -> None:
    with pytest.raises(UsageError, match="subset"):
        select_files([], section, "everything")


def test_listing_follows_page_tokens() -> None:
    bucket = FakeBucket({object_name(f"s{i}"): b"x" for i in range(7)}, page_size=3)
    items = list_objects(bucket, LISTING, "split-videos/testing/16FPS/", retries=0)
    assert len(items) == 7
    assert [p.get("pageToken") for p in bucket.listings] == [None, "3", "6"]


# --- Verified, resumable download -------------------------------------------------------------


def test_a_cut_download_resumes_from_the_part_file(tmp_path: Path) -> None:
    data = bytes(range(256)) * 4
    name = object_name("ball-ramp")
    bucket = FakeBucket({name: data})
    bucket.cut[name] = 400
    outcome = fetch_file(bucket, remote(name, data), tmp_path, retries=2, sleep=no_sleep)
    assert outcome.path.read_bytes() == data
    assert (outcome.transferred, outcome.cached) == (len(data), False)
    assert bucket.ranges == [(name, 0), (name, 400)]
    assert not part_path(outcome.path).exists()


def test_a_leftover_part_is_continued(tmp_path: Path) -> None:
    data = b"0123456789" * 50
    name = object_name("ball-ramp")
    part = part_path(tmp_path / name)
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:120])
    bucket = FakeBucket({name: data})
    outcome = fetch_file(bucket, remote(name, data), tmp_path, retries=0, sleep=no_sleep)
    assert bucket.ranges == [(name, 120)]
    assert outcome.transferred == len(data) - 120
    assert outcome.path.read_bytes() == data


def test_a_bad_checksum_is_never_promoted(tmp_path: Path) -> None:
    data = b"not the published bytes"
    name = object_name("ball-ramp")
    bucket = FakeBucket({name: data})
    with pytest.raises(ChecksumError, match="md5"):
        fetch_file(bucket, remote(name, data, md5=md5_of(b"other")), tmp_path, 0, no_sleep)
    target = tmp_path / name
    assert not target.exists()
    assert not part_path(target).exists()


def test_an_intact_file_is_not_downloaded_again(tmp_path: Path) -> None:
    data = b"cached"
    name = object_name("ball-ramp")
    (tmp_path / name).parent.mkdir(parents=True)
    (tmp_path / name).write_bytes(data)
    bucket = FakeBucket({name: data})
    outcome = fetch_file(bucket, remote(name, data), tmp_path, retries=0, sleep=no_sleep)
    assert outcome.cached and outcome.transferred == 0
    assert bucket.calls == 0


# --- Dataset ----------------------------------------------------------------------------------


def fetched(
    settings: Settings, section: PhysicsIQSettings, mp4_bytes: bytes
) -> tuple[PhysicsIQDataset, FakeBucket]:
    bucket = FakeBucket(local_objects(section, mp4_bytes), page_size=4)
    dataset = PhysicsIQDataset(settings, client=bucket)
    dataset.fetch("local", settings.paths.raw_dir / dataset.name)
    return dataset, bucket


def test_fetch_writes_a_manifest_and_a_second_run_transfers_nothing(
    settings: Settings, section: PhysicsIQSettings, mp4_bytes: bytes
) -> None:
    dataset, bucket = fetched(settings, section, mp4_bytes)
    dest = settings.paths.raw_dir / dataset.name
    manifest = read_manifest(dest / MANIFEST_FILE)
    assert manifest is not None
    assert (manifest.dataset, manifest.subset, manifest.license) == (
        "physics_iq",
        "local",
        "CC-BY-4.0",
    )
    assert manifest.source == BUCKET
    assert len(manifest.clips) == 5 and len(manifest.files) == 10
    assert {f.md5_b64 for f in manifest.files} == {md5_of(mp4_bytes)}
    assert len(bucket.ranges) == 10
    again = FakeBucket({})
    PhysicsIQDataset(settings, client=again).fetch("local", dest)
    assert again.calls == 0


def test_clips_use_the_container_rate_and_cap_the_frames(
    settings: Settings, section: PhysicsIQSettings, mp4_bytes: bytes
) -> None:
    dataset, _ = fetched(settings, section, mp4_bytes)
    clips = dataset.clips("local")
    assert [c.clip_id for c in clips] == [f"{s}__center_take1" for s in section.local]
    clip = clips[0]
    assert clip.time_source is TimeSource.CONTAINER_FPS
    assert clip.container_fps == 16.0
    assert clip.seconds_per_frame == pytest.approx(1 / 16)
    assert clip.n_frames == MAX_FRAMES
    assert (clip.width, clip.height) == SIZE
    assert clip.scene_id == "ball-and-block-fall"
    assert dataset.ground_truth(clip) is None
    assert dataset.video(clip).n_frames == N_FRAMES
    assert [dataset.is_static(c) for c in clips] == [False, False, False, False, True]


def test_motion_masks_follow_the_sampling(
    settings: Settings, section: PhysicsIQSettings, mp4_bytes: bytes
) -> None:
    dataset, _ = fetched(settings, section, mp4_bytes)
    clip = dataset.clips("local")[0]
    sampling = FrameSampling(
        frame_indices=np.arange(4, dtype=np.int64),
        width=SIZE[0],
        height=SIZE[1],
        native_width=SIZE[0],
        native_height=SIZE[1],
        seconds_per_frame=1 / 16,
        step=1,
        max_side=None,
    )
    grey = dataset.motion_masks(clip, sampling)
    assert grey.shape == (4, SIZE[1], SIZE[0]) and grey.dtype == np.uint8
    assert grey[3].mean() > grey[0].mean()
    wrong = FrameSampling(**{**sampling.__dict__, "width": SIZE[0] * 2})
    with pytest.raises(MissingInputError, match="mask size"):
        dataset.motion_masks(clip, wrong)


def test_annotations_by_clip_id_then_by_scenario(
    settings: Settings, section: PhysicsIQSettings, mp4_bytes: bytes
) -> None:
    table = {
        "balls-collide": [{"phrase": "ball", "expected_count": 2}],
        "balls-collide__center_take1": [{"phrase": "red ball", "expected_count": 2}],
        "ball-rolls-off": [{"phrase": "ball", "expected_count": 1}],
    }
    path = settings.paths.annotations_dir / "physics_iq.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table), encoding="utf-8")
    dataset, _ = fetched(settings, section, mp4_bytes)
    queries = {c.scene_id: c.object_queries for c in dataset.clips("local")}
    assert queries["balls-collide"] == (ObjectQuery(phrase="red ball", expected_count=2),)
    assert queries["ball-rolls-off"] == (ObjectQuery(phrase="ball", expected_count=1),)
    assert queries["stable-blocks"] == ()


def test_clips_before_fetch_point_to_the_fetch_command(settings: Settings) -> None:
    with pytest.raises(MissingInputError, match="data fetch physics_iq"):
        PhysicsIQDataset(settings).clips("local")


def test_repo_annotations_cover_every_configured_scenario(section: PhysicsIQSettings) -> None:
    table = load_annotations(REPO_ANNOTATIONS / "physics_iq.json")
    scenarios = {*section.local, *section.full_dynamic, *section.full_static}
    assert scenarios <= set(table)
    assert all(table[s] for s in scenarios)
    assert load_annotations(REPO_ANNOTATIONS / "pisa_real.json") == {}
