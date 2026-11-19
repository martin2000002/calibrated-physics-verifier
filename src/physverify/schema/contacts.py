"""Contact manifolds (not single points: R2 balances torques) and raw simulator collisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from physverify.schema._validate import (
    ArraySpec,
    check_bool,
    check_enum,
    check_finite,
    check_instance,
    check_number,
    check_optional_finite,
    check_str,
    check_unit_vector,
)
from physverify.schema.constants import ENV_ID, MAX_MANIFOLD_POINTS
from physverify.schema.errors import SchemaError
from physverify.schema.io import Record
from physverify.schema.states import CoordFrame, LengthUnit

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from numpy.typing import NDArray


class ManifoldKind(str, Enum):
    """``measured``: simulator points. ``primitive``: primitive geometry + poses.
    ``point``: only one reliable point (R2 abstains on it)."""

    MEASURED = "measured"
    PRIMITIVE = "primitive"
    POINT = "point"


class ContactKind(str, Enum):
    IMPACT = "impact"
    RESTING = "resting"


@dataclass(frozen=True, eq=False)
class ContactRecord(Record):
    """One object pair over one interval with an approximately constant contact patch.

    ``normal`` points from ``object_b`` to ``object_a`` (the force on a acts along +normal).
    ``points`` are the patch vertices in the frame and unit of the table; ``t_start``/``t_end``
    are fractional native frame indices; negative ``point_distances`` mean penetration.
    """

    object_a: str
    object_b: str
    t_start: float
    t_end: float
    kind: ContactKind
    normal: NDArray[np.float64]
    points: NDArray[np.float64]
    manifold: ManifoldKind
    point_normals: NDArray[np.float64] | None = None
    point_distances: NDArray[np.float64] | None = None
    orientation_assumed: bool = False
    max_penetration: float | None = None
    peak_force: float | None = None
    confidence: float = math.nan
    source: str = ""

    _arrays = MappingProxyType(
        {
            "normal": ArraySpec("f8", (3,)),
            "points": ArraySpec("f8", ("K", 3)),
            "point_normals": ArraySpec("f8", ("K", 3), optional=True),
            "point_distances": ArraySpec("f8", ("K",), optional=True),
        }
    )

    def validate(self) -> None:
        super().validate()
        check_str("object_a", self.object_a)
        check_str("object_b", self.object_b)
        if self.object_a == ENV_ID or self.object_a == self.object_b:
            raise SchemaError("object_a must be a movable object distinct from object_b")
        check_finite("t_start", self.t_start)
        check_finite("t_end", self.t_end)
        if self.t_start > self.t_end:
            raise SchemaError("t_start must be <= t_end")
        check_enum("kind", self.kind, ContactKind)
        check_enum("manifold", self.manifold, ManifoldKind)
        check_unit_vector("normal", self.normal, allow_nan=False)
        self._validate_points()
        check_bool("orientation_assumed", self.orientation_assumed)
        check_optional_finite("max_penetration", self.max_penetration)
        check_optional_finite("peak_force", self.peak_force)
        _check_confidence(self.confidence)
        check_str("source", self.source, allow_empty=True)

    def _validate_points(self) -> None:
        count = self.points.shape[0]
        if not 1 <= count <= MAX_MANIFOLD_POINTS:
            raise SchemaError(f"a manifold has 1..{MAX_MANIFOLD_POINTS} points, got {count}")
        if self.manifold is ManifoldKind.POINT and count != 1:
            raise SchemaError("manifold=point must have exactly one point")
        if not bool(np.all(np.isfinite(self.points))):
            raise SchemaError("contact points must be finite")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            object_a=data["object_a"],
            object_b=data["object_b"],
            t_start=data["t_start"],
            t_end=data["t_end"],
            kind=ContactKind(data["kind"]),
            manifold=ManifoldKind(data["manifold"]),
            orientation_assumed=data["orientation_assumed"],
            max_penetration=data["max_penetration"],
            peak_force=data["peak_force"],
            confidence=data["confidence"],
            source=data["source"],
            **cls.decode_arrays(data),
        )


def _check_confidence(value: object) -> None:
    number = check_number("confidence", value)
    if not (math.isnan(number) or 0.0 <= number <= 1.0):
        raise SchemaError("confidence must lie in [0, 1] or be NaN")


@dataclass(frozen=True, eq=False)
class ContactTable(Record):
    """All contacts of a clip, in the frame, unit and ``source_key`` of its LIFT table."""

    clip_id: str
    frame: CoordFrame
    unit: LengthUnit
    source_key: str
    records: tuple[ContactRecord, ...]

    def validate(self) -> None:
        check_str("clip_id", self.clip_id)
        check_enum("frame", self.frame, CoordFrame)
        check_enum("unit", self.unit, LengthUnit)
        check_str("source_key", self.source_key)
        if not isinstance(self.records, tuple):
            raise SchemaError("records must be a tuple")
        for record in self.records:
            check_instance("records", record, ContactRecord)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            clip_id=data["clip_id"],
            frame=CoordFrame(data["frame"]),
            unit=LengthUnit(data["unit"]),
            source_key=data["source_key"],
            records=tuple(ContactRecord.from_dict(r) for r in data["records"]),
        )


@dataclass(frozen=True, eq=False)
class RawCollisions(Record):
    """Simulator collision events, one row per point and substep (world frame, meters)."""

    scene_id: str
    a: NDArray[np.str_]
    b: NDArray[np.str_]
    frame: NDArray[np.float64]
    position: NDArray[np.float64]
    normal: NDArray[np.float64]
    force: NDArray[np.float64]

    _arrays = MappingProxyType(
        {
            "a": ArraySpec("U", ("M",)),
            "b": ArraySpec("U", ("M",)),
            "frame": ArraySpec("f8", ("M",)),
            "position": ArraySpec("f8", ("M", 3)),
            "normal": ArraySpec("f8", ("M", 3)),
            "force": ArraySpec("f8", ("M",)),
        }
    )

    def validate(self) -> None:
        super().validate()
        check_str("scene_id", self.scene_id)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(scene_id=data["scene_id"], **cls.decode_arrays(data))
