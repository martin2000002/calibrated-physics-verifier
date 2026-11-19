"""Scene specifications consumed by the Kubric worker and the perturbation record."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from physverify.schema._validate import (
    check_bool,
    check_enum,
    check_finite,
    check_float_tuple,
    check_instance,
    check_int,
    check_json_mapping,
    check_positive,
    check_span,
    check_str,
    check_str_tuple,
)
from physverify.schema.constants import STANDARD_GRAVITY
from physverify.schema.errors import SchemaError
from physverify.schema.io import Record, optional
from physverify.schema.states import PrimitiveKind

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self


class AssetSource(str, Enum):
    KUBASIC = "kubasic"
    GSO = "gso"


class MagnitudeKind(str, Enum):
    """``check_unit``: magnitude in the unit of the target check, computed with physics.measures.
    ``injected_parameter``: the injected parameter itself, in its own unit (R2 overhang)."""

    CHECK_UNIT = "check_unit"
    INJECTED_PARAMETER = "injected_parameter"


def _floats(values: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


@dataclass(frozen=True, eq=False)
class CameraSpec(Record):
    """Fixed pinhole camera in world coordinates (z up)."""

    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    focal_length_mm: float = 35.0
    sensor_width_mm: float = 32.0

    def validate(self) -> None:
        check_float_tuple("position", self.position, 3)
        check_float_tuple("look_at", self.look_at, 3)
        check_positive("focal_length_mm", self.focal_length_mm)
        check_positive("sensor_width_mm", self.sensor_width_mm)
        if tuple(self.position) == tuple(self.look_at):
            raise SchemaError("camera position and look_at must differ")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            position=_floats(data["position"]),  # type: ignore[arg-type]
            look_at=_floats(data["look_at"]),  # type: ignore[arg-type]
            focal_length_mm=data["focal_length_mm"],
            sensor_width_mm=data["sensor_width_mm"],
        )


@dataclass(frozen=True, eq=False)
class ObjectSpec(Record):
    """One rigid object. ``extents`` are full extents in meters along the local axes; the
    conversion to Kubric scale (half-extent, radius) happens only in kubric_worker/build_scene.py.
    ``label`` is the noun phrase used as the text query for detection."""

    object_id: str
    label: str
    asset_source: AssetSource
    asset_id: str
    primitive: PrimitiveKind
    extents: tuple[float, float, float]
    mass: float
    friction: float
    restitution: float
    static: bool
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: tuple[float, float, float] = (0.5, 0.5, 0.5)

    def validate(self) -> None:
        for label in ("object_id", "label", "asset_id"):
            check_str(label, getattr(self, label))
        check_enum("asset_source", self.asset_source, AssetSource)
        check_enum("primitive", self.primitive, PrimitiveKind)
        check_float_tuple("extents", self.extents, 3)
        for extent in self.extents:
            check_positive("extents", extent)
        check_positive("mass", self.mass)
        for label in ("friction", "restitution"):
            check_finite(label, getattr(self, label))
        check_bool("static", self.static)
        for label, length in (("position", 3), ("quaternion", 4), ("velocity", 3)):
            check_float_tuple(label, getattr(self, label), length)
        check_float_tuple("angular_velocity", self.angular_velocity, 3)
        check_float_tuple("color", self.color, 3)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        names = ("extents", "position", "quaternion", "velocity", "angular_velocity", "color")
        vectors: dict[str, Any] = {name: _floats(data[name]) for name in names}
        return cls(
            object_id=data["object_id"],
            label=data["label"],
            asset_source=AssetSource(data["asset_source"]),
            asset_id=data["asset_id"],
            primitive=PrimitiveKind(data["primitive"]),
            mass=data["mass"],
            friction=data["friction"],
            restitution=data["restitution"],
            static=data["static"],
            **vectors,
        )


@dataclass(frozen=True, eq=False)
class SceneSpec(Record):
    """Everything the worker needs to simulate and render one scene."""

    scene_id: str
    family: str
    seed: int
    n_frames: int
    settle_frames: int
    resolution: tuple[int, int]
    camera: CameraSpec
    objects: tuple[ObjectSpec, ...]
    frame_rate: int = 24
    step_rate: int = 240
    gravity: tuple[float, float, float] = (0.0, 0.0, -STANDARD_GRAVITY)
    spp: int = 64
    background: str = "plain"

    def validate(self) -> None:
        check_str("scene_id", self.scene_id)
        check_str("family", self.family)
        check_int("seed", self.seed, minimum=0)
        check_int("n_frames", self.n_frames, minimum=1)
        check_int("settle_frames", self.settle_frames, minimum=0)
        self._validate_resolution()
        check_instance("camera", self.camera, CameraSpec)
        self._validate_objects()
        check_int("frame_rate", self.frame_rate, minimum=1)
        check_int("step_rate", self.step_rate, minimum=1)
        if self.step_rate % self.frame_rate:
            raise SchemaError("step_rate must be a multiple of frame_rate")
        check_float_tuple("gravity", self.gravity, 3)
        check_int("spp", self.spp, minimum=1)
        check_str("background", self.background)

    def _validate_resolution(self) -> None:
        if not isinstance(self.resolution, tuple) or len(self.resolution) != 2:
            raise SchemaError("resolution must be (width, height)")
        for size in self.resolution:
            check_int("resolution", size, minimum=1)

    def _validate_objects(self) -> None:
        if not isinstance(self.objects, tuple):
            raise SchemaError("objects must be a tuple of ObjectSpec")
        for obj in self.objects:
            check_instance("objects", obj, ObjectSpec)
        ids = [obj.object_id for obj in self.objects]
        if len(set(ids)) != len(ids):
            raise SchemaError("object ids must be unique")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            scene_id=data["scene_id"],
            family=data["family"],
            seed=data["seed"],
            n_frames=data["n_frames"],
            settle_frames=data["settle_frames"],
            resolution=tuple(data["resolution"]),
            camera=CameraSpec.from_dict(data["camera"]),
            objects=tuple(ObjectSpec.from_dict(obj) for obj in data["objects"]),
            frame_rate=data["frame_rate"],
            step_rate=data["step_rate"],
            gravity=_floats(data["gravity"]),  # type: ignore[arg-type]
            spp=data["spp"],
            background=data["background"],
        )


@dataclass(frozen=True, eq=False)
class PerturbationRecord(Record):
    """What was injected into a clip and how large it is (see MagnitudeKind)."""

    name: str
    target_checks: tuple[str, ...]
    magnitude: float
    magnitude_unit: str
    magnitude_kind: MagnitudeKind
    level: int
    target_objects: tuple[str, ...]
    frame_span: tuple[int, int] | None
    seed: int
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        check_str("name", self.name)
        check_str_tuple("target_checks", self.target_checks)
        check_finite("magnitude", self.magnitude)
        check_str("magnitude_unit", self.magnitude_unit)
        check_enum("magnitude_kind", self.magnitude_kind, MagnitudeKind)
        check_int("level", self.level, minimum=0)
        check_str_tuple("target_objects", self.target_objects)
        check_int("seed", self.seed, minimum=0)
        check_json_mapping("params", self.params)
        if self.frame_span is not None:
            check_span("frame_span", self.frame_span)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            name=data["name"],
            target_checks=tuple(data["target_checks"]),
            magnitude=data["magnitude"],
            magnitude_unit=data["magnitude_unit"],
            magnitude_kind=MagnitudeKind(data["magnitude_kind"]),
            level=data["level"],
            target_objects=tuple(data["target_objects"]),
            frame_span=optional(data["frame_span"], tuple),
            seed=data["seed"],
            params=data["params"],
        )
