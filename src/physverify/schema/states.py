"""Canonical per-object state table shared by ground truth and perception."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from physverify.schema._validate import (
    ArraySpec,
    check_enum,
    check_sigma,
    check_sigma_array,
    check_str,
    check_unit_vector,
)
from physverify.schema.errors import SchemaError
from physverify.schema.io import Record

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Self

    from numpy.typing import NDArray


class CoordFrame(str, Enum):
    """``world_zup``: simulator world, z up. ``camera_opencv``: x right, y down, z forward."""

    WORLD_ZUP = "world_zup"
    CAMERA_OPENCV = "camera_opencv"


class LengthUnit(str, Enum):
    METER = "meter"
    RELATIVE = "relative"


class PrimitiveKind(str, Enum):
    """Collision primitive; geometry treats ``unknown`` as a box."""

    SPHERE = "sphere"
    BOX = "box"
    CYLINDER = "cylinder"
    UNKNOWN = "unknown"


_PRIMITIVE_VALUES = frozenset(kind.value for kind in PrimitiveKind)

_VECTOR_FIELDS = (
    "position",
    "position_sigma",
    "velocity",
    "velocity_sigma",
    "acceleration",
    "acceleration_sigma",
    "angular_velocity",
    "size",
    "size_sigma",
)
_SIGMA_FIELDS = (
    "position_sigma",
    "orientation_sigma",
    "velocity_sigma",
    "acceleration_sigma",
    "size_sigma",
)


def ground_truth_source_key(clip_id: str) -> str:
    """``source_key`` of ground-truth tables (the oracle path of the scale rule)."""
    return f"gt:{clip_id}"


@dataclass(frozen=True, eq=False)
class ObjectStateTable(Record):
    """Structure of arrays over T frames x N objects. NaN means not available.

    ``size`` holds full extents along the local axes (or the assumed axes when ``quaternion`` is
    NaN). ``source_key`` is the cache key of the LIFT output that fixed geometry and unit.
    """

    clip_id: str
    frame: CoordFrame
    unit: LengthUnit
    source_key: str
    t: NDArray[np.float64]
    frame_index: NDArray[np.int64]
    object_ids: NDArray[np.str_]
    labels: NDArray[np.str_]
    primitive: NDArray[np.str_]
    position: NDArray[np.float64]
    position_sigma: NDArray[np.float64]
    quaternion: NDArray[np.float64]
    orientation_sigma: NDArray[np.float64]
    velocity: NDArray[np.float64]
    velocity_sigma: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    acceleration_sigma: NDArray[np.float64]
    angular_velocity: NDArray[np.float64]
    size: NDArray[np.float64]
    size_sigma: NDArray[np.float64]
    visible: NDArray[np.bool_]
    in_frame: NDArray[np.bool_]
    occlusion: NDArray[np.float64]
    confidence: NDArray[np.float64]
    gravity_dir: NDArray[np.float64]
    gravity_dir_sigma: float

    _arrays = MappingProxyType(
        {
            "t": ArraySpec("f8", ("T",)),
            "frame_index": ArraySpec("i8", ("T",)),
            "object_ids": ArraySpec("U", ("N",)),
            "labels": ArraySpec("U", ("N",)),
            "primitive": ArraySpec("U", ("N",)),
            **{name: ArraySpec("f8", ("T", "N", 3)) for name in _VECTOR_FIELDS},
            "quaternion": ArraySpec("f8", ("T", "N", 4)),
            "orientation_sigma": ArraySpec("f8", ("T", "N")),
            "visible": ArraySpec("?", ("T", "N")),
            "in_frame": ArraySpec("?", ("T", "N")),
            "occlusion": ArraySpec("f8", ("T", "N")),
            "confidence": ArraySpec("f8", ("T", "N")),
            "gravity_dir": ArraySpec("f8", (3,)),
        }
    )

    @property
    def n_frames(self) -> int:
        return int(self.t.shape[0])

    @property
    def n_objects(self) -> int:
        return int(self.object_ids.shape[0])

    def object_index(self, object_id: str) -> int:
        matches = np.flatnonzero(self.object_ids == object_id)
        if matches.size != 1:
            raise KeyError(object_id)
        return int(matches[0])

    def validate(self) -> None:
        super().validate()
        check_str("clip_id", self.clip_id)
        check_enum("frame", self.frame, CoordFrame)
        check_enum("unit", self.unit, LengthUnit)
        check_str("source_key", self.source_key)
        check_sigma("gravity_dir_sigma", self.gravity_dir_sigma)
        check_unit_vector("gravity_dir", self.gravity_dir)
        self._validate_ids()
        self._validate_time()
        for name in _SIGMA_FIELDS:
            check_sigma_array(name, getattr(self, name))
        for name in ("occlusion", "confidence"):
            _check_unit_interval(name, getattr(self, name))

    def _validate_ids(self) -> None:
        if len(set(self.object_ids.tolist())) != self.n_objects:
            raise SchemaError("object_ids must be unique")
        unknown = set(self.primitive.tolist()) - _PRIMITIVE_VALUES
        if unknown:
            raise SchemaError(f"unknown primitive kinds {sorted(unknown)}")

    def _validate_time(self) -> None:
        if not bool(np.all(np.isfinite(self.t))) or bool(np.any(np.diff(self.t) <= 0)):
            raise SchemaError("t must be finite and strictly increasing")

    @classmethod
    def blank(
        cls,
        *,
        clip_id: str,
        frame: CoordFrame,
        unit: LengthUnit,
        source_key: str,
        t: NDArray[np.float64],
        frame_index: NDArray[np.int64],
        object_ids: Sequence[str],
        labels: Sequence[str],
        primitive: Sequence[str],
    ) -> Self:
        """A table with every measured quantity NaN, visible and in frame everywhere."""
        n_t, n_obj = int(t.shape[0]), len(object_ids)
        vectors: dict[str, Any] = {
            name: np.full((n_t, n_obj, 3), np.nan) for name in _VECTOR_FIELDS
        }
        return cls(
            clip_id=clip_id,
            frame=frame,
            unit=unit,
            source_key=source_key,
            t=np.asarray(t, dtype=np.float64),
            frame_index=np.asarray(frame_index, dtype=np.int64),
            object_ids=np.array(list(object_ids), dtype=str),
            labels=np.array(list(labels), dtype=str),
            primitive=np.array(list(primitive), dtype=str),
            **vectors,
            quaternion=np.full((n_t, n_obj, 4), np.nan),
            orientation_sigma=np.full((n_t, n_obj), np.nan),
            visible=np.ones((n_t, n_obj), dtype=bool),
            in_frame=np.ones((n_t, n_obj), dtype=bool),
            occlusion=np.full((n_t, n_obj), np.nan),
            confidence=np.full((n_t, n_obj), np.nan),
            gravity_dir=np.full(3, np.nan),
            gravity_dir_sigma=float("nan"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            clip_id=data["clip_id"],
            frame=CoordFrame(data["frame"]),
            unit=LengthUnit(data["unit"]),
            source_key=data["source_key"],
            gravity_dir_sigma=data["gravity_dir_sigma"],
            **cls.decode_arrays(data),
        )


def _check_unit_interval(label: str, values: NDArray[np.float64]) -> None:
    finite = values[~np.isnan(values)]
    if bool(np.any(finite < 0)) or bool(np.any(finite > 1)):
        raise SchemaError(f"{label}: values must lie in [0, 1] or be NaN")
