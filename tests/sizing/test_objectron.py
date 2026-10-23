"""ADR 0010 I4: the wire decoder reads Objectron bytes built here (no network)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from physverify.sizing import objectron, protobuf_wire
from physverify.sizing.objectron import ObjectronError, VideoRef
from physverify.sizing.samples import samples_of


def varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def tag(number: int, wire_type: int) -> bytes:
    return varint(number << 3 | wire_type)


def length(number: int, payload: bytes) -> bytes:
    return tag(number, protobuf_wire.LENGTH) + varint(len(payload)) + payload


def packed(number: int, values: list[float]) -> bytes:
    return length(number, struct.pack(f"<{len(values)}f", *values))


def single_float(number: int, value: float) -> bytes:
    return tag(number, protobuf_wire.FIXED32) + struct.pack("<f", value)


def integer(number: int, value: int) -> bytes:
    return tag(number, protobuf_wire.VARINT) + varint(value)


def box(object_id: int, category: str, extents: list[float]) -> bytes:
    return (
        integer(objectron.OBJECT_ID, object_id)
        + length(objectron.OBJECT_CATEGORY, category.encode())
        + integer(3, 1)
        + packed(4, [1.0] * 9)
        + packed(objectron.OBJECT_SCALE, extents)
    )


def keypoint(index: int, x: float, y: float, depth: float) -> bytes:
    point = single_float(1, x) + single_float(2, y) + single_float(3, depth)
    return integer(objectron.KEYPOINT_ID, index) + length(objectron.KEYPOINT_2D, point)


def frame(frame_id: int, depth: float) -> bytes:
    annotation = integer(objectron.ANNOTATION_OBJECT, 0) + length(
        objectron.ANNOTATION_KEYPOINTS, keypoint(0, 0.5, 0.25, depth)
    )
    camera = (
        integer(objectron.CAMERA_WIDTH, 1920)
        + integer(objectron.CAMERA_HEIGHT, 1440)
        + packed(objectron.CAMERA_INTRINSICS, [1500.0, 0, 960, 0, 1500.0, 720, 0, 0, 1])
    )
    timestamp = tag(4, protobuf_wire.FIXED64) + struct.pack("<d", frame_id / 30)
    return (
        integer(objectron.FRAME_ID, frame_id)
        + length(objectron.FRAME_ANNOTATIONS, annotation)
        + length(objectron.FRAME_CAMERA, camera)
        + timestamp
    )


CUP_EXTENTS = [0.07, 0.11, 0.075]
SEQUENCE = (
    length(objectron.SEQUENCE_OBJECTS, box(0, "chair", CUP_EXTENTS))
    + length(objectron.SEQUENCE_OBJECTS, box(1, "chair", [1.0, 1.0, 1.0]))
    + length(objectron.SEQUENCE_FRAMES, frame(0, 0.42))
    + length(objectron.SEQUENCE_FRAMES, frame(1, 0.44))
)


def test_varints_round_trip() -> None:
    for value in (0, 1, 127, 128, 300, 2**35 + 7):
        assert protobuf_wire.read_varint(varint(value), 0) == (value, len(varint(value)))


def test_category_scale_and_center_depth_decode() -> None:
    parsed = objectron.sequence_objects(SEQUENCE, complete=True)
    assert parsed is not None
    boxes, end = parsed
    assert [b.category for b in boxes] == ["chair", "chair"]
    np.testing.assert_allclose(boxes[0].extents, CUP_EXTENTS, rtol=1e-6)
    frames = objectron.frame_records(SEQUENCE[end:])
    assert [f.frame_id for f in frames] == [0, 1]
    center = frames[1].objects[0]
    assert center.depth_m[objectron.CENTER_KEYPOINT] == pytest.approx(0.44, rel=1e-6)
    np.testing.assert_allclose(center.points_2d[0], [0.5, 0.25], rtol=1e-6)
    assert (frames[0].width, frames[0].height) == (1920, 1440)
    assert frames[0].intrinsics[0] == pytest.approx(1500.0)


def test_a_prefix_is_undecided_until_the_objects_end() -> None:
    parsed = objectron.sequence_objects(SEQUENCE, complete=True)
    assert parsed is not None
    end = parsed[1]
    for cut in range(1, end + 1):
        assert objectron.sequence_objects(SEQUENCE[:cut], complete=False) is None
    prefix = objectron.sequence_objects(SEQUENCE[: end + 1], complete=False)
    assert prefix is not None and prefix[1] == end
    with pytest.raises(protobuf_wire.WireError):
        objectron.sequence_objects(SEQUENCE[: end - 3], complete=True)


def test_the_label_comes_from_the_directory_and_placeholders_are_dropped() -> None:
    parsed = objectron.sequence_objects(SEQUENCE, complete=True)
    assert parsed is not None
    ref = VideoRef.from_index_line("cup/batch-16/47\n")
    samples = samples_of(ref, parsed[0], "test", min_extent_m=0.01)
    assert [(s.label, s.video_id) for s in samples] == [("cup", "cup/batch-16/47")]
    assert samples_of(ref, parsed[0], "test", min_extent_m=0.08) == []


def test_bad_index_lines_and_groups_are_rejected() -> None:
    with pytest.raises(ObjectronError):
        VideoRef.from_index_line("../etc/passwd")
    with pytest.raises(protobuf_wire.WireError):
        protobuf_wire.read_field(tag(1, 3), 0)


class FakeReader:
    """Serves ``SEQUENCE`` by ranges and records the requested sizes."""

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def read_text(self, url: str) -> str:
        return "cup/batch-1/0\ncup/batch-1/1\ncup/batch-2/5\n"

    def read_prefix(self, url: str, size: int) -> tuple[bytes, bool]:
        self.sizes.append(size)
        return SEQUENCE[:size], size >= len(SEQUENCE)


def test_heads_grow_until_whole_and_are_stored_trimmed(tmp_path: Path) -> None:
    reader = FakeReader()
    refs = objectron.read_index(reader, "https://bucket", "cup", "test", tmp_path / "index")
    assert [r.video_id for r in refs] == ["cup/batch-1/0", "cup/batch-1/1", "cup/batch-2/5"]
    plan = objectron.HeadFetch("https://bucket", tmp_path / "heads", 16, retries=0)
    objectron.fetch_heads(reader, plan, refs[:1], workers=1)
    assert reader.sizes[:3] == [16, 32, 64]
    stored = objectron.load_head(objectron.head_path(plan.heads, refs[0]))
    np.testing.assert_allclose(stored[0].extents, CUP_EXTENTS, rtol=1e-6)


def test_selection_is_seeded_per_class() -> None:
    refs = [VideoRef(f"cup/batch-1/{i}") for i in range(50)]
    first = objectron.select(refs, "cup", 10, seed=0)
    assert first == objectron.select(list(reversed(refs)), "cup", 10, seed=0)
    assert first != objectron.select(refs, "cup", 10, seed=1)
    assert len(set(first)) == 10


def test_a_video_in_both_splits_is_refused() -> None:
    train = [VideoRef("cup/batch-1/0"), VideoRef("cup/batch-1/1")]
    objectron.require_disjoint(train, [VideoRef("cup/batch-2/0")])
    with pytest.raises(ObjectronError, match="1 videos in both splits"):
        objectron.require_disjoint(train, [VideoRef("cup/batch-1/1")])


class CacheOnly:
    def read_text(self, url: str) -> str:
        raise AssertionError(f"not cached: {url}")

    def read_prefix(self, url: str, size: int) -> tuple[bytes, bool]:
        raise AssertionError(f"not cached: {url}")


INDEX_CACHE = Path(__file__).resolve().parents[2] / "data" / "raw" / "objectron" / "index"


@pytest.mark.parametrize(
    "label", ["bike", "book", "bottle", "camera", "cereal_box", "chair", "cup", "laptop", "shoe"]
)
def test_the_official_train_and_test_indices_share_no_video(label: str) -> None:
    """I5 on the real (cached, git-ignored) index: no fitted video is a test video."""
    splits = [INDEX_CACHE / f"{label}_annotations_{split}" for split in objectron.SPLITS]
    if not all(path.is_file() for path in splits):
        pytest.skip("Objectron index not fetched (run `physverify size fetch`)")
    train, test = (
        objectron.read_index(CacheOnly(), "unused", label, split, INDEX_CACHE)
        for split in objectron.SPLITS
    )
    assert train and test
    objectron.require_disjoint(train, test)
