"""Objectron annotations: index, ranged download of box sizes, per-frame keypoint depth.

Schema (``objectron/schema/{annotation_data,object,a_r_capture_metadata}.proto``): ``Sequence``
1 objects, 2 frame_annotations; ``Object`` 1 id, 2 category, 6 scale (box extents, meters);
``FrameAnnotation`` 1 frame_id, 2 annotations, 3 camera; ``ObjectAnnotation`` 1 object_id,
2 keypoints; ``AnnotatedKeyPoint`` 1 id, 3 point_2d (x 1, y 2 normalized, depth 3 meters);
``ARCamera`` 5 width, 6 height, 7 intrinsics (3x3 row-major). Keypoint 0 is the box center.
"""

from __future__ import annotations

import functools
import re
import urllib.request
import zlib
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from physverify.core.atomic import write_text_atomic
from physverify.core.download import TIMEOUT_S, retrying
from physverify.core.errors import PhysverifyError
from physverify.sizing import protobuf_wire

SEQUENCE_OBJECTS = 1
SEQUENCE_FRAMES = 2
OBJECT_ID, OBJECT_CATEGORY, OBJECT_SCALE = 1, 2, 6
FRAME_ID, FRAME_ANNOTATIONS, FRAME_CAMERA = 1, 2, 3
ANNOTATION_OBJECT, ANNOTATION_KEYPOINTS = 1, 2
KEYPOINT_ID, KEYPOINT_2D = 1, 3
POINT_X, POINT_Y, POINT_DEPTH = 1, 2, 3
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_INTRINSICS = 5, 6, 7
CENTER_KEYPOINT = 0
BOX_KEYPOINTS = 9
SPLITS = ("train", "test")
HEAD_SUFFIX = ".head"
VIDEO_FILE = "video.MOV"
_INDEX_LINE = re.compile(r"^[a-z_]+/batch-\d+/\d+$")
_CONTENT_RANGE_TOTAL = re.compile(r"/(\d+)$")


class ObjectronError(PhysverifyError):
    """An Objectron index line or annotation that cannot be read."""


@dataclass(frozen=True)
class VideoRef:
    """One annotated video; the class comes from its directory, never from ``category``."""

    video_id: str

    @property
    def label(self) -> str:
        return self.video_id.split("/", 1)[0]

    @classmethod
    def from_index_line(cls, line: str) -> VideoRef:
        text = line.strip()
        if not _INDEX_LINE.match(text):
            raise ObjectronError(f"not an Objectron index line: {line!r}")
        return cls(text)


def index_url(bucket_url: str, label: str, split: str) -> str:
    return f"{bucket_url}/v1/index/{label}_annotations_{split}"


def annotation_url(bucket_url: str, ref: VideoRef) -> str:
    return f"{bucket_url}/annotations/{ref.video_id}.pbdata"


def video_prefix(ref: VideoRef) -> str:
    return f"videos/{ref.video_id}/"


@dataclass(frozen=True)
class ObjectBox:
    object_id: int
    category: str
    extents: tuple[float, float, float]


def parse_object(payload: bytes) -> ObjectBox:
    fields = list(protobuf_wire.iter_fields(payload))
    scale = protobuf_wire.repeated_floats(fields, OBJECT_SCALE)
    if len(scale) != 3:
        raise ObjectronError(f"object scale has {len(scale)} values, expected 3")
    ident = protobuf_wire.first(fields, OBJECT_ID)
    category = protobuf_wire.first(fields, OBJECT_CATEGORY)
    return ObjectBox(
        object_id=0 if ident is None else ident.integer,
        category="" if category is None else category.payload.decode("utf-8", "replace"),
        extents=(scale[0], scale[1], scale[2]),
    )


def sequence_objects(buf: bytes, *, complete: bool) -> tuple[list[ObjectBox], int] | None:
    """The boxes of a ``Sequence`` (or of a prefix of one) and the byte where they end."""
    leading = protobuf_wire.leading_records(buf, SEQUENCE_OBJECTS, complete=complete)
    if leading is None:
        return None
    records, end = leading
    return [parse_object(record) for record in records], end


# --- Per-frame keypoints (end-to-end truth) -----------------------------------------------------


@dataclass(frozen=True)
class FrameObject:
    object_id: int
    points_2d: np.ndarray
    """(9, 2) keypoints in normalized image coordinates (x right, y down, in [0, 1])."""
    depth_m: np.ndarray
    """(9,) metric depth of each keypoint from ARKit."""


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    objects: tuple[FrameObject, ...]
    intrinsics: tuple[float, ...]
    width: int
    height: int


def frame_records(buf: bytes) -> list[FrameRecord]:
    """Every ``FrameAnnotation`` of a complete ``Sequence``."""
    return [
        _frame(field.payload)
        for field in protobuf_wire.iter_fields(buf)
        if field.number == SEQUENCE_FRAMES
    ]


def _frame(payload: bytes) -> FrameRecord:
    fields = list(protobuf_wire.iter_fields(payload))
    ident = protobuf_wire.first(fields, FRAME_ID)
    camera = protobuf_wire.first(fields, FRAME_CAMERA)
    camera_fields = [] if camera is None else list(protobuf_wire.iter_fields(camera.payload))
    width, height = (
        protobuf_wire.first(camera_fields, CAMERA_WIDTH),
        protobuf_wire.first(camera_fields, CAMERA_HEIGHT),
    )
    return FrameRecord(
        frame_id=0 if ident is None else ident.integer,
        objects=tuple(_frame_object(f.payload) for f in fields if f.number == FRAME_ANNOTATIONS),
        intrinsics=protobuf_wire.repeated_floats(camera_fields, CAMERA_INTRINSICS),
        width=0 if width is None else width.integer,
        height=0 if height is None else height.integer,
    )


def _frame_object(payload: bytes) -> FrameObject:
    fields = list(protobuf_wire.iter_fields(payload))
    ident = protobuf_wire.first(fields, ANNOTATION_OBJECT)
    points = np.full((BOX_KEYPOINTS, 2), np.nan)
    depth = np.full(BOX_KEYPOINTS, np.nan)
    for field in fields:
        if field.number == ANNOTATION_KEYPOINTS:
            _keypoint_into(field.payload, points, depth)
    return FrameObject(0 if ident is None else ident.integer, points, depth)


def _keypoint_into(payload: bytes, points: np.ndarray, depth: np.ndarray) -> None:
    fields = list(protobuf_wire.iter_fields(payload))
    ident = protobuf_wire.first(fields, KEYPOINT_ID)
    point = protobuf_wire.first(fields, KEYPOINT_2D)
    index = 0 if ident is None else ident.integer
    if point is None or not 0 <= index < BOX_KEYPOINTS:
        return
    coords = list(protobuf_wire.iter_fields(point.payload))
    values = {n: protobuf_wire.repeated_floats(coords, n) for n in (POINT_X, POINT_Y, POINT_DEPTH)}
    points[index] = [_scalar(values[POINT_X]), _scalar(values[POINT_Y])]
    depth[index] = _scalar(values[POINT_DEPTH])


def _scalar(values: Sequence[float]) -> float:
    return values[0] if values else float("nan")


# --- Network ----------------------------------------------------------------------------------


class PrefixReader(Protocol):
    def read_text(self, url: str) -> str: ...

    def read_prefix(self, url: str, size: int) -> tuple[bytes, bool]:
        """The first ``size`` bytes of ``url`` (fewer at its end) and whether that is all of it."""
        ...


class UrllibPrefixReader:
    def __init__(self, timeout_s: float = TIMEOUT_S) -> None:
        self._timeout_s = timeout_s

    def read_text(self, url: str) -> str:
        with urllib.request.urlopen(url, timeout=self._timeout_s) as response:
            text: str = response.read().decode("utf-8")
        return text

    def read_prefix(self, url: str, size: int) -> tuple[bytes, bool]:
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{size - 1}"})
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            body: bytes = response.read()
            total = _CONTENT_RANGE_TOTAL.search(response.headers.get("Content-Range") or "")
        return body, total is None or int(total.group(1)) <= len(body)


def read_index(
    reader: PrefixReader, bucket_url: str, label: str, split: str, cache: Path
) -> list[VideoRef]:
    """The official split of ``label`` (cached as text under ``cache``)."""
    path = cache / f"{label}_annotations_{split}"
    if not path.is_file():
        write_text_atomic(path, reader.read_text(index_url(bucket_url, label, split)))
    lines = path.read_text(encoding="utf-8").splitlines()
    return [VideoRef.from_index_line(line) for line in lines if line.strip()]


def require_disjoint(train: Iterable[VideoRef], test: Iterable[VideoRef]) -> None:
    """Refuse a train and a test split that share a video (no test object may be fitted on)."""
    shared = sorted({r.video_id for r in train} & {r.video_id for r in test})
    if shared:
        raise ObjectronError(f"{len(shared)} videos in both splits, e.g. {shared[:3]}")


def select(refs: Sequence[VideoRef], label: str, count: int, seed: int) -> list[VideoRef]:
    """``count`` videos drawn without replacement, reproducible per (seed, label)."""
    ordered = sorted(refs, key=lambda ref: ref.video_id)
    rng = np.random.default_rng([seed, zlib.crc32(label.encode())])
    chosen = rng.choice(len(ordered), size=min(count, len(ordered)), replace=False)
    return [ordered[i] for i in sorted(chosen.tolist())]


def head_path(heads: Path, ref: VideoRef) -> Path:
    return heads / f"{ref.video_id}{HEAD_SUFFIX}"


def fetch_head(
    reader: PrefixReader, url: str, start_bytes: int, retries: int
) -> tuple[bytes, list[ObjectBox]]:
    """The leading ``Sequence.objects`` bytes, doubling the ranged read until they are whole."""
    size = start_bytes
    while True:
        request = functools.partial(reader.read_prefix, url, size)
        body, complete = retrying(request, retries, f"read of {url}")
        parsed = sequence_objects(body, complete=complete)
        if parsed is not None:
            boxes, end = parsed
            return body[:end], boxes
        if complete:
            raise ObjectronError(f"{url}: truncated annotation")
        size *= 2


def load_head(path: Path) -> list[ObjectBox]:
    parsed = sequence_objects(path.read_bytes(), complete=True)
    if parsed is None:
        raise ObjectronError(f"{path}: unreadable head")
    return parsed[0]


@dataclass(frozen=True)
class HeadFetch:
    bucket_url: str
    heads: Path
    start_bytes: int
    retries: int


def fetch_heads(
    reader: PrefixReader,
    plan: HeadFetch,
    refs: Iterable[VideoRef],
    workers: int,
    progress: Callable[[VideoRef], None] | None = None,
) -> int:
    """Download every missing head; returns the bytes transferred."""

    def one(ref: VideoRef) -> int:
        path = head_path(plan.heads, ref)
        if path.is_file():
            return 0
        body, _ = fetch_head(
            reader, annotation_url(plan.bucket_url, ref), plan.start_bytes, plan.retries
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_bytes(body)
        part.replace(path)
        if progress is not None:
            progress(ref)
        return len(body)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(one, list(refs)))
