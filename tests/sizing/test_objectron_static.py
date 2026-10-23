"""``objectron_static`` (plan-congelado 3.2, F3, J14): the preregistered selection over fabricated
Objectron indices and a fake bucket; no real video is downloaded."""

from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np
import pytest

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.core.download import MANIFEST_FILE, intact, read_manifest
from physverify.core.errors import ConfigError, MissingInputError, UsageError
from physverify.datasets.objectron_static import (
    FAILED,
    FRAMES_DIR,
    SELECTION_FILE,
    TAKEN,
    TOO_LARGE,
    Decoded,
    ObjectronStaticDataset,
    ObjectronStaticSettings,
    decode_upright,
    order_key,
    ordered,
    read_selection,
)
from physverify.schema.clip import ObjectQuery, Split, TimeSource
from physverify.sizing import objectron
from physverify.sizing.objectron_draws import e2e_draw
from physverify.sizing.settings import sizing_settings

BUCKET = "https://bucket.test/objectron"
LISTING = "https://listing.test/o"
CLASSES = ("cup", "cereal_box")
VIDEOS_PER_SPLIT = 12
PER_CLASS = 2
A9_TEST = 2
SMALL, LARGE = 10, 100
FRAMES = 3
FPS = 30.0


def md5_of(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode("ascii")


def ids_of(label: str, split: str) -> list[str]:
    batch = 1 if split == "test" else 2
    return [f"{label}/batch-{batch}/{i}" for i in range(VIDEOS_PER_SPLIT)]


def mov_name(video_id: str) -> str:
    return f"videos/{video_id}/video.MOV"


class FakeReader:
    """The official indices as ``read_index`` fetches them."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    def read_text(self, url: str) -> str:
        self.reads.append(url)
        stem = url.rsplit("/", 1)[-1]
        label, split = stem.split("_annotations_")
        return "\n".join(ids_of(label, split)) + "\n"

    def read_prefix(self, url: str, size: int) -> tuple[bytes, bool]:
        raise AssertionError("objectron_static never reads annotation heads")


class FakeBucket:
    """GCS listing by exact object name and ranged GETs; ``corrupt`` bodies fail their md5."""

    def __init__(self, objects: Mapping[str, bytes]) -> None:
        self.objects = dict(objects)
        self.corrupt: set[str] = set()
        self.gets: list[str] = []
        self.listings = 0

    def get_json(self, url: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        self.listings += 1
        names = sorted(n for n in self.objects if n.startswith(params["prefix"]))
        items = [
            {"name": n, "size": str(len(self.objects[n])), "md5Hash": md5_of(self.objects[n])}
            for n in names
        ]
        return {"items": items}

    def open_range(self, url: str, start: int) -> tuple[int, BinaryIO]:
        name = url.removeprefix(f"{BUCKET}/")
        self.gets.append(name)
        body = self.objects[name][start:]
        if name in self.corrupt:
            body = bytes(len(body))
        return (206 if start else 200), io.BytesIO(body)


class FakeDecoder:
    """Writes ``FRAMES`` small JPEG frames; ``broken`` videos decode to none."""

    def __init__(self) -> None:
        self.broken: set[str] = set()
        self.calls: list[Path] = []

    def __call__(self, video: Path, target: Path, seconds: float, max_side: int) -> Decoded:
        self.calls.append(video)
        if any(video.as_posix().endswith(mov_name(v)) for v in self.broken):
            return Decoded(FPS, 0, 0, 0)
        target.mkdir(parents=True, exist_ok=True)
        for i in range(FRAMES):
            cv2.imwrite(str(target / f"{i:05d}.jpg"), np.full((8, 6, 3), 10 * i, np.uint8))
        return Decoded(FPS, FRAMES, 6, 8)


@pytest.fixture
def make(config_dir: Path, environ: dict[str, str]) -> Callable[..., Settings]:
    def build(*overrides: str, env: Mapping[str, str] | None = None) -> Settings:
        base = (
            f'sizing.bucket_url="{BUCKET}"',
            f'sizing.listing_url="{LISTING}"',
            f"sizing.classes={list(CLASSES)}",
            f"sizing.per_class_test={A9_TEST}",
            "sizing.per_class_train=2",
            'sizing.e2e.classes=["cup"]',
            "sizing.e2e.videos=1",
            f"sizing.e2e.max_video_mb={SMALL * 2 / 1e6}",
            f"objectron_static.per_class={PER_CLASS}",
            f"objectron_static.max_video_mb={SMALL * 2 / 1e6}",
        )
        return load_settings(
            config_dir, environ={**environ, **(env or {})}, overrides=(*base, *overrides)
        )

    return build


@pytest.fixture
def settings(make: Callable[..., Settings]) -> Settings:
    return make()


@pytest.fixture
def bucket() -> FakeBucket:
    return FakeBucket(
        {mov_name(v): v.encode().ljust(SMALL, b"-") for c in CLASSES for v in ids_of(c, "test")}
    )


@pytest.fixture
def decoder() -> FakeDecoder:
    return FakeDecoder()


def dataset_of(
    settings: Settings, bucket: FakeBucket, decoder: FakeDecoder
) -> ObjectronStaticDataset:
    return ObjectronStaticDataset(settings, client=bucket, reader=FakeReader(), decoder=decoder)


def fetch(settings: Settings, bucket: FakeBucket, decoder: FakeDecoder) -> Path:
    dest = settings.paths.raw_dir / "objectron_static"
    dataset_of(settings, bucket, decoder).fetch("full", dest)
    return dest


def a9_looked_at(settings: Settings) -> set[str]:
    """The A9 test draw plus the first end-to-end candidate of ``cup``, recomputed by hand."""
    sizing = sizing_settings(settings)
    refs = {c: [objectron.VideoRef(v) for v in ids_of(c, "test")] for c in CLASSES}
    drawn = {r.video_id for c in CLASSES for r in objectron.select(refs[c], c, A9_TEST, 0)}
    e2e_first = objectron.select(refs["cup"], "cup", VIDEOS_PER_SPLIT, sizing.seed)[0]
    return drawn | {e2e_first.video_id}


def expected_order(settings: Settings, label: str) -> list[str]:
    seed = settings.section("objectron_static", ObjectronStaticSettings).seed
    excluded = frozenset(a9_looked_at(settings))
    refs = [objectron.VideoRef(v) for v in ids_of(label, "test")]
    return [r.video_id for r in ordered(refs, excluded, seed)]


# --- Rule ---------------------------------------------------------------------------------------


def test_the_order_is_the_sha256_of_seed_and_id() -> None:
    refs = [objectron.VideoRef(v) for v in ids_of("cup", "test")]
    order = [r.video_id for r in ordered(refs, frozenset({"cup/batch-1/3"}), 7)]
    assert "cup/batch-1/3" not in order and len(order) == VIDEOS_PER_SPLIT - 1
    assert order == sorted(order, key=lambda v: hashlib.sha256(f"7:{v}".encode()).hexdigest())
    assert order_key(7, "cup/batch-1/0") == hashlib.sha256(b"7:cup/batch-1/0").hexdigest()
    assert order != [r.video_id for r in ordered(refs, frozenset({"cup/batch-1/3"}), 8)]


def test_the_plugin_is_registered_and_the_freeze_seed_comes_from_the_environment(
    make: Callable[..., Settings],
) -> None:
    assert PLUGIN_KINDS["datasets"]()["objectron_static"] is ObjectronStaticDataset
    env = {"PHYSVERIFY__OBJECTRON_STATIC__SEED": "1234"}
    assert make(env=env).section("objectron_static", ObjectronStaticSettings).seed == 1234


def test_the_e2e_draw_replays_size_e2e_prepare(settings: Settings) -> None:
    sizing = sizing_settings(settings)
    plan = sizing.e2e.model_copy(update={"classes": CLASSES, "videos": 3})
    refs = [objectron.VideoRef(v) for v in ids_of("cup", "test")]
    order = objectron.select(refs, "cup", VIDEOS_PER_SPLIT, sizing.seed)
    draw = e2e_draw(settings, sizing, plan, FakeReader(), lambda r: r != order[0])
    assert [r.label for r in draw.taken] == ["cup", "cup", "cereal_box"]
    assert draw.skipped == (order[0],)
    assert draw.taken[:2] == tuple(order[1:3])
    assert draw.examined == {r.video_id for r in (*draw.taken, order[0])}


# --- Fetch --------------------------------------------------------------------------------------


def test_fetch_leaves_out_every_video_a9_looked_at(
    settings: Settings, bucket: FakeBucket, decoder: FakeDecoder
) -> None:
    dest = fetch(settings, bucket, decoder)
    chosen = read_selection(dest / SELECTION_FILE)
    assert chosen is not None
    assert set(chosen.excluded) == a9_looked_at(settings)
    assert not {c.video_id for c in chosen.clips} & set(chosen.excluded)
    assert all("/batch-1/" in c.video_id for c in chosen.clips)


def test_fetch_takes_the_first_per_class_of_the_order(
    settings: Settings, bucket: FakeBucket, decoder: FakeDecoder
) -> None:
    dest = fetch(settings, bucket, decoder)
    chosen = read_selection(dest / SELECTION_FILE)
    assert chosen is not None
    for label in CLASSES:
        taken = [c.video_id for c in chosen.clips if c.label == label]
        assert taken == expected_order(settings, label)[:PER_CLASS]
    manifest = read_manifest(dest / MANIFEST_FILE)
    assert manifest is not None and manifest.dataset == "objectron_static"
    assert manifest.license == "C-UDA-1.0"
    assert manifest.clips == tuple(c.clip_id for c in chosen.clips)
    assert all(f.sha256 and intact(dest / f.relative, f) for f in manifest.files)
    assert all((dest / FRAMES_DIR / c.clip_id / "00000.jpg").is_file() for c in chosen.clips)


def test_a_large_failed_or_undecodable_video_gives_way_to_the_next(
    settings: Settings, bucket: FakeBucket, decoder: FakeDecoder
) -> None:
    first, second, third, fourth, fifth = expected_order(settings, "cup")[:5]
    bucket.objects[mov_name(first)] = b"x" * LARGE
    bucket.corrupt.add(mov_name(second))
    decoder.broken.add(third)
    dest = fetch(settings, bucket, decoder)
    chosen = read_selection(dest / SELECTION_FILE)
    assert chosen is not None
    cup = [(c.video_id, c.status) for c in chosen.candidates if c.video_id.startswith("cup/")]
    assert cup == [
        (first, TOO_LARGE),
        (second, FAILED),
        (third, FAILED),
        (fourth, TAKEN),
        (fifth, TAKEN),
    ]
    assert mov_name(first) not in bucket.gets
    assert not (dest / mov_name(second)).exists() and not (dest / mov_name(third)).exists()
    assert not (dest / FRAMES_DIR / third.replace("/", "_")).exists()


def test_the_configured_split_is_the_one_drawn_and_recorded(
    make: Callable[..., Settings], decoder: FakeDecoder
) -> None:
    """A dry run passes ``split = train`` and never touches the sealed test videos (B1)."""
    settings = make(env={"PHYSVERIFY__OBJECTRON_STATIC__SPLIT": "train"})
    train = FakeBucket(
        {mov_name(v): v.encode().ljust(SMALL, b"-") for c in CLASSES for v in ids_of(c, "train")}
    )
    dest = fetch(settings, train, decoder)
    chosen = read_selection(dest / SELECTION_FILE)
    assert chosen is not None and chosen.split == "train"
    assert len(chosen.clips) == PER_CLASS * len(CLASSES)
    assert all("/batch-2/" in c.video_id for c in chosen.clips)
    assert all("/batch-2/" in name for name in train.gets)
    sizing = sizing_settings(settings)
    refs = {c: [objectron.VideoRef(v) for v in ids_of(c, "train")] for c in CLASSES}
    a9_train = {r.video_id for c in CLASSES for r in objectron.select(refs[c], c, 2, sizing.seed)}
    assert a9_train <= set(chosen.excluded)
    assert not {c.video_id for c in chosen.clips} & set(chosen.excluded)
    manifest = read_manifest(dest / MANIFEST_FILE)
    assert manifest is not None and "official train index" in manifest.revision
    with pytest.raises(UsageError, match="fixed once made"):
        fetch(make(), train, decoder)


def test_the_configured_split_is_test_and_anything_else_is_rejected(
    make: Callable[..., Settings],
) -> None:
    assert make().section("objectron_static", ObjectronStaticSettings).split == "test"
    with pytest.raises(ConfigError, match="split"):
        make(env={"PHYSVERIFY__OBJECTRON_STATIC__SPLIT": "validation"}).section(
            "objectron_static", ObjectronStaticSettings
        )


def test_a_second_fetch_transfers_nothing(
    settings: Settings, bucket: FakeBucket, decoder: FakeDecoder
) -> None:
    fetch(settings, bucket, decoder)
    again = FakeBucket({})
    fetch(settings, again, decoder)
    assert again.listings == 0 and again.gets == []


def test_another_seed_is_refused_once_a_selection_exists(
    make: Callable[..., Settings], bucket: FakeBucket, decoder: FakeDecoder
) -> None:
    fetch(make(), bucket, decoder)
    other = make("objectron_static.seed=99")
    with pytest.raises(UsageError, match="fixed once made"):
        fetch(other, bucket, decoder)


def test_lost_files_are_restored_never_replaced(
    settings: Settings, bucket: FakeBucket, decoder: FakeDecoder
) -> None:
    dest = fetch(settings, bucket, decoder)
    before = (dest / SELECTION_FILE).read_text(encoding="utf-8")
    chosen = read_selection(dest / SELECTION_FILE)
    assert chosen is not None
    lost = chosen.clips[0]
    (dest / mov_name(lost.video_id)).unlink()
    fetch(settings, bucket, decoder)
    assert (dest / mov_name(lost.video_id)).is_file()
    assert (dest / SELECTION_FILE).read_text(encoding="utf-8") == before
    (dest / mov_name(lost.video_id)).unlink()
    bucket.corrupt.add(mov_name(lost.video_id))
    with pytest.raises(MissingInputError, match="cannot be restored"):
        fetch(settings, bucket, decoder)
    assert (dest / SELECTION_FILE).read_text(encoding="utf-8") == before


# --- Clips --------------------------------------------------------------------------------------


def test_clips_are_eval_with_the_class_query_and_no_count(
    settings: Settings, bucket: FakeBucket, decoder: FakeDecoder
) -> None:
    fetch(settings, bucket, decoder)
    dataset = dataset_of(settings, bucket, decoder)
    clips = dataset.clips("full")
    assert len(clips) == PER_CLASS * len(CLASSES)
    assert [c.clip_id for c in dataset.clips("local")] == [c.clip_id for c in clips]
    assert [c.clip_id for c in dataset.clips(None)] == [c.clip_id for c in clips]
    cereal = next(c for c in clips if c.clip_id.startswith("cereal_box_"))
    assert cereal.object_queries == (ObjectQuery(phrase="cereal box", expected_count=None),)
    assert cereal.scene_id == cereal.clip_id.replace("_batch", "/batch").replace("-1_", "-1/")
    assert cereal.split is Split.EVAL and cereal.time_source is TimeSource.CONTAINER_FPS
    assert cereal.container_fps == FPS and cereal.real_fps == pytest.approx(FPS)
    assert (cereal.n_frames, cereal.width, cereal.height) == (FRAMES, 6, 8)
    assert cereal.license == "C-UDA-1.0" and dataset.ground_truth(cereal) is None
    video = dataset.video(cereal)
    assert video.n_frames == FRAMES and video.native_size == (6, 8)


def test_clips_before_fetch_and_unknown_subsets_are_refused(settings: Settings) -> None:
    dataset = ObjectronStaticDataset(settings)
    with pytest.raises(MissingInputError, match="data fetch objectron_static"):
        dataset.clips("full")
    with pytest.raises(UsageError, match="unknown objectron_static subset"):
        dataset.clips("everything")


def test_decode_upright_keeps_the_native_rate(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (16, 12))
    for i in range(30):
        writer.write(np.full((12, 16, 3), 5 * i, np.uint8))
    writer.release()
    decoded = decode_upright(video, tmp_path / "frames", 0.5, 640)
    assert decoded == Decoded(FPS, 15, 16, 12)
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 15
