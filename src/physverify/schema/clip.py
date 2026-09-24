"""Clip manifests, text queries and frame sampling (request and resolved form)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from physverify.schema._validate import (
    ArraySpec,
    check_enum,
    check_instance,
    check_int,
    check_optional_str,
    check_positive,
    check_str,
)
from physverify.schema.errors import SchemaError
from physverify.schema.io import Record, optional
from physverify.schema.scene import PerturbationRecord
from physverify.schema.stages import Stage

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from numpy.typing import NDArray


class Split(str, Enum):
    CALIB = "calib"
    EVAL = "eval"
    ALL = "all"


class TimeSource(str, Enum):
    """Where ``seconds_per_frame`` comes from (PISA declares 1/120 s)."""

    GROUND_TRUTH = "ground_truth"
    CONTAINER_FPS = "container_fps"
    DECLARED = "declared"


@dataclass(frozen=True, eq=False)
class ObjectQuery(Record):
    """Noun phrase for the text-conditioned detector, with the expected count when known."""

    phrase: str
    expected_count: int | None = None

    def validate(self) -> None:
        check_str("phrase", self.phrase)
        if self.expected_count is not None:
            check_int("expected_count", self.expected_count, minimum=1)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(phrase=data["phrase"], expected_count=data["expected_count"])


@dataclass(frozen=True, eq=False)
class ClipManifest(Record):
    """One clip of a dataset. ``video_path`` is relative to the data directory."""

    dataset: str
    clip_id: str
    split: Split
    video_path: str
    n_frames: int
    width: int
    height: int
    container_fps: float
    seconds_per_frame: float
    time_source: TimeSource
    object_queries: tuple[ObjectQuery, ...] = ()
    scene_id: str | None = None
    perturbation: PerturbationRecord | None = None
    gt_stages: tuple[Stage, ...] = ()
    prompt: str | None = None
    license: str = ""

    @property
    def real_fps(self) -> float:
        """Frames per second of real (physical) time, not of the container."""
        return 1.0 / self.seconds_per_frame

    def validate(self) -> None:
        for label in ("dataset", "clip_id", "video_path"):
            check_str(label, getattr(self, label))
        check_enum("split", self.split, Split)
        check_int("n_frames", self.n_frames, minimum=1)
        check_int("width", self.width, minimum=1)
        check_int("height", self.height, minimum=1)
        check_positive("container_fps", self.container_fps)
        check_positive("seconds_per_frame", self.seconds_per_frame)
        check_enum("time_source", self.time_source, TimeSource)
        for query in _as_tuple("object_queries", self.object_queries):
            check_instance("object_queries", query, ObjectQuery)
        check_optional_str("scene_id", self.scene_id)
        if self.perturbation is not None:
            check_instance("perturbation", self.perturbation, PerturbationRecord)
        for stage in _as_tuple("gt_stages", self.gt_stages):
            check_enum("gt_stages", stage, Stage)
        check_optional_str("prompt", self.prompt)
        check_str("license", self.license, allow_empty=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            dataset=data["dataset"],
            clip_id=data["clip_id"],
            split=Split(data["split"]),
            video_path=data["video_path"],
            n_frames=data["n_frames"],
            width=data["width"],
            height=data["height"],
            container_fps=data["container_fps"],
            seconds_per_frame=data["seconds_per_frame"],
            time_source=TimeSource(data["time_source"]),
            object_queries=tuple(ObjectQuery.from_dict(q) for q in data["object_queries"]),
            scene_id=data["scene_id"],
            perturbation=optional(data["perturbation"], PerturbationRecord.from_dict),
            gt_stages=tuple(Stage(s) for s in data["gt_stages"]),
            prompt=data["prompt"],
            license=data["license"],
        )


def _as_tuple(label: str, value: object) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise SchemaError(f"{label}: expected a tuple")
    return value


@dataclass(frozen=True, eq=False)
class SamplingSpec(Record):
    """Requested sampling. ``target_fps`` is in real time; None keeps every native frame."""

    target_fps: float | None = None
    max_side: int | None = None
    start: int = 0
    stop: int | None = None

    def validate(self) -> None:
        if self.target_fps is not None:
            check_positive("target_fps", self.target_fps)
        if self.max_side is not None:
            check_int("max_side", self.max_side, minimum=1)
        check_int("start", self.start, minimum=0)
        if self.stop is not None:
            check_int("stop", self.stop, minimum=self.start + 1)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            target_fps=data["target_fps"],
            max_side=data["max_side"],
            start=data["start"],
            stop=data["stop"],
        )


@dataclass(frozen=True, eq=False)
class FrameSampling(Record):
    """Resolved sampling: native frame indices and the processing size in pixels."""

    frame_indices: NDArray[np.int64]
    width: int
    height: int
    native_width: int
    native_height: int
    seconds_per_frame: float
    step: int
    max_side: int | None

    _arrays = MappingProxyType({"frame_indices": ArraySpec("i8", ("T",))})

    @property
    def n_frames(self) -> int:
        return int(self.frame_indices.shape[0])

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def native_size(self) -> tuple[int, int]:
        return (self.native_width, self.native_height)

    @property
    def start(self) -> int:
        return int(self.frame_indices[0])

    @property
    def stop(self) -> int:
        """Exclusive native stop index, as accepted by ``VideoSource.frames``."""
        return int(self.frame_indices[-1]) + 1

    def times(self) -> NDArray[np.float64]:
        """Real time in seconds of every sampled frame."""
        times: NDArray[np.float64] = self.frame_indices.astype(np.float64) * self.seconds_per_frame
        return times

    def validate(self) -> None:
        super().validate()
        for label in ("width", "height", "native_width", "native_height", "step"):
            check_int(label, getattr(self, label), minimum=1)
        check_positive("seconds_per_frame", self.seconds_per_frame)
        if self.max_side is not None:
            check_int("max_side", self.max_side, minimum=1)
        if self.n_frames == 0:
            raise SchemaError("sampling must select at least one frame")
        if bool(np.any(np.diff(self.frame_indices) <= 0)) or self.start < 0:
            raise SchemaError("frame_indices must be non-negative and strictly increasing")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            width=data["width"],
            height=data["height"],
            native_width=data["native_width"],
            native_height=data["native_height"],
            seconds_per_frame=data["seconds_per_frame"],
            step=data["step"],
            max_side=data["max_side"],
            **cls.decode_arrays(data),
        )


def round_half_up(value: float) -> int:
    """Round to the nearest integer with halves going up (Python's round() is banker's)."""
    return int(math.floor(value + 0.5))


def scaled_size(width: int, height: int, max_side: int | None) -> tuple[int, int]:
    """Processing size: the longer side is at most ``max_side``; never upsampled."""
    factor = 1.0 if max_side is None else min(1.0, max_side / max(width, height))
    return (max(1, round_half_up(width * factor)), max(1, round_half_up(height * factor)))


def frame_step(real_fps: float, target_fps: float | None) -> int:
    return 1 if target_fps is None else max(1, round_half_up(real_fps / target_fps))


def resolve_sampling(clip: ClipManifest, spec: SamplingSpec) -> FrameSampling:
    """Turn a request into native frame indices and a processing size for this clip."""
    stop = clip.n_frames if spec.stop is None else min(spec.stop, clip.n_frames)
    if spec.start >= stop:
        raise SchemaError(f"{clip.clip_id}: start {spec.start} is past the last frame")
    step = frame_step(clip.real_fps, spec.target_fps)
    width, height = scaled_size(clip.width, clip.height, spec.max_side)
    return FrameSampling(
        frame_indices=np.arange(spec.start, stop, step, dtype=np.int64),
        width=width,
        height=height,
        native_width=clip.width,
        native_height=clip.height,
        seconds_per_frame=clip.seconds_per_frame,
        step=step,
        max_side=spec.max_side,
    )
