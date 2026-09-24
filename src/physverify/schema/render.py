"""Raw render layers written by the worker (Kubric has no writer for the z layer)."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from physverify.schema._validate import ArraySpec, check_str
from physverify.schema.errors import SchemaError
from physverify.schema.io import Record

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self

    from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class RenderLayers(Record):
    """Native-resolution layers of a rendered clip.

    ``segmentation`` value 0 is background and value k > 0 is ``object_ids[k - 1]``; ``z`` is
    the camera-space depth in meters.
    """

    clip_id: str
    frame_indices: NDArray[np.int64]
    rgba: NDArray[np.uint8]
    segmentation: NDArray[np.uint32]
    z: NDArray[np.float32]
    object_ids: NDArray[np.str_]

    _arrays = MappingProxyType(
        {
            "frame_indices": ArraySpec("i8", ("T",)),
            "rgba": ArraySpec("u1", ("T", "H", "W", 4)),
            "segmentation": ArraySpec("u4", ("T", "H", "W")),
            "z": ArraySpec("f4", ("T", "H", "W")),
            "object_ids": ArraySpec("U", ("N",)),
        }
    )

    def validate(self) -> None:
        super().validate()
        check_str("clip_id", self.clip_id)
        if self.segmentation.size and int(self.segmentation.max()) > self.object_ids.shape[0]:
            raise SchemaError("segmentation refers to an object index outside object_ids")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(clip_id=data["clip_id"], **cls.decode_arrays(data))
