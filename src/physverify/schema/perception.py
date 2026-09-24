"""Pixel-space stage outputs. Each carries the FrameSampling it was computed on."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from physverify.schema._validate import (
    ArraySpec,
    check_arrays,
    check_enum,
    check_instance,
    check_int,
    check_number,
    check_sigma,
)
from physverify.schema.clip import FrameSampling
from physverify.schema.errors import SchemaError
from physverify.schema.io import Record, decode_optional_array

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from numpy.typing import NDArray


class DepthKind(str, Enum):
    METRIC_Z = "metric_z"
    RELATIVE_DEPTH = "relative_depth"
    AFFINE_DISPARITY = "affine_disparity"


class IntrinsicsSource(str, Enum):
    GROUND_TRUTH = "ground_truth"
    ASSUMED_FOV = "assumed_fov"
    ESTIMATED = "estimated"


def _check_sampling(sampling: object, frames: int | None, height: int, width: int) -> None:
    """Pixel arrays must match the processing size (and frame count) of their sampling."""
    check_instance("sampling", sampling, FrameSampling)
    assert isinstance(sampling, FrameSampling)
    if (height, width) != (sampling.height, sampling.width):
        raise SchemaError(
            f"arrays are {width}x{height} but sampling is {sampling.width}x{sampling.height}"
        )
    if frames is not None and frames != sampling.n_frames:
        raise SchemaError(f"arrays have {frames} frames but sampling has {sampling.n_frames}")


@dataclass(frozen=True, eq=False)
class Detections(Record):
    """Boxes on the first sampled frame, in processing pixels (x0, y0, x1, y1)."""

    sampling: FrameSampling
    frame_index: int
    boxes_xyxy: NDArray[np.float64]
    scores: NDArray[np.float64]
    labels: NDArray[np.str_]
    object_ids: NDArray[np.str_]

    _arrays = MappingProxyType(
        {
            "boxes_xyxy": ArraySpec("f8", ("K", 4)),
            "scores": ArraySpec("f8", ("K",)),
            "labels": ArraySpec("U", ("K",)),
            "object_ids": ArraySpec("U", ("K",)),
        }
    )

    def validate(self) -> None:
        super().validate()
        check_instance("sampling", self.sampling, FrameSampling)
        check_int("frame_index", self.frame_index, minimum=0)
        if self.frame_index not in self.sampling.frame_indices.tolist():
            raise SchemaError("frame_index must be one of the sampled frames")
        if len(set(self.object_ids.tolist())) != self.object_ids.shape[0]:
            raise SchemaError("object_ids must be unique")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            sampling=FrameSampling.from_dict(data["sampling"]),
            frame_index=data["frame_index"],
            **cls.decode_arrays(data),
        )


@dataclass(frozen=True, eq=False)
class MaskTrack(Record):
    """Per-object masks over the sampled frames (T', N, h, w)."""

    sampling: FrameSampling
    object_ids: NDArray[np.str_]
    labels: NDArray[np.str_]
    masks: NDArray[np.bool_]
    scores: NDArray[np.float64]

    _arrays = MappingProxyType(
        {
            "object_ids": ArraySpec("U", ("N",)),
            "labels": ArraySpec("U", ("N",)),
            "masks": ArraySpec("?", ("T", "N", "h", "w")),
            "scores": ArraySpec("f8", ("T", "N")),
        }
    )

    def validate(self) -> None:
        super().validate()
        frames, _, height, width = self.masks.shape
        _check_sampling(self.sampling, frames, height, width)
        if len(set(self.object_ids.tolist())) != self.object_ids.shape[0]:
            raise SchemaError("object_ids must be unique")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(sampling=FrameSampling.from_dict(data["sampling"]), **cls.decode_arrays(data))


@dataclass(frozen=True, eq=False)
class DepthMap(Record):
    """Raw depth prediction (no sigma: heavy stages store raw outputs)."""

    sampling: FrameSampling
    depth: NDArray[np.float32]
    kind: DepthKind
    valid: NDArray[np.bool_]

    _arrays = MappingProxyType(
        {
            "depth": ArraySpec("f4", ("T", "h", "w")),
            "valid": ArraySpec("?", ("T", "h", "w")),
        }
    )

    def validate(self) -> None:
        super().validate()
        check_enum("kind", self.kind, DepthKind)
        frames, height, width = self.depth.shape
        _check_sampling(self.sampling, frames, height, width)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            sampling=FrameSampling.from_dict(data["sampling"]),
            kind=DepthKind(data["kind"]),
            **cls.decode_arrays(data),
        )


@dataclass(frozen=True, eq=False)
class CameraModel(Record):
    """OpenCV intrinsics in processing pixels; ``world_from_camera`` is (4,4), (T',4,4) or None.

    Intrinsics use the continuous image-plane convention, so resizing scales every entry of K.
    """

    sampling: FrameSampling
    K: NDArray[np.float64]
    source: IntrinsicsSource
    fov_deg: float
    fov_sigma_deg: float
    world_from_camera: NDArray[np.float64] | None = None

    _arrays = MappingProxyType({"K": ArraySpec("f8", (3, 3))})

    def validate(self) -> None:
        super().validate()
        check_instance("sampling", self.sampling, FrameSampling)
        check_enum("source", self.source, IntrinsicsSource)
        check_number("fov_deg", self.fov_deg)
        check_sigma("fov_sigma_deg", self.fov_sigma_deg)
        if not bool(np.all(np.isfinite(self.K))):
            raise SchemaError("K must be finite")
        if self.world_from_camera is not None:
            self._validate_pose()

    def _validate_pose(self) -> None:
        pose = self.world_from_camera
        shape = ("T", 4, 4) if isinstance(pose, np.ndarray) and pose.ndim == 3 else (4, 4)
        check_arrays(
            "CameraModel",
            {"world_from_camera": pose},
            {"world_from_camera": ArraySpec("f8", shape)},
        )
        assert pose is not None
        if pose.ndim == 3 and pose.shape[0] != self.sampling.n_frames:
            raise SchemaError("per-frame world_from_camera must have one pose per sampled frame")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(
            sampling=FrameSampling.from_dict(data["sampling"]),
            source=IntrinsicsSource(data["source"]),
            fov_deg=data["fov_deg"],
            fov_sigma_deg=data["fov_sigma_deg"],
            world_from_camera=decode_optional_array(data.get("world_from_camera")),
            **cls.decode_arrays(data),
        )
