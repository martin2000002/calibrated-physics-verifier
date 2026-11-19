"""Output type of every stage (separate module to keep stages.py free of import cycles)."""

from __future__ import annotations

from types import MappingProxyType
from typing import Union

from physverify.schema.contacts import ContactTable
from physverify.schema.errors import SchemaError
from physverify.schema.io import Record
from physverify.schema.perception import CameraModel, DepthMap, Detections, MaskTrack
from physverify.schema.scale import ScaleEstimate
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

StageOutput = Union[
    Detections, MaskTrack, DepthMap, CameraModel, ObjectStateTable, ScaleEstimate, ContactTable
]

STAGE_OUTPUT_TYPES: MappingProxyType[Stage, type[Record]] = MappingProxyType(
    {
        Stage.DETECT: Detections,
        Stage.TRACK: MaskTrack,
        Stage.DEPTH: DepthMap,
        Stage.CAMERA: CameraModel,
        Stage.LIFT: ObjectStateTable,
        Stage.SCALE: ScaleEstimate,
        Stage.KINEMATICS: ObjectStateTable,
        Stage.CONTACTS: ContactTable,
    }
)


def check_stage_output(stage: Stage, output: object) -> None:
    expected = STAGE_OUTPUT_TYPES[stage]
    if not isinstance(output, expected):
        raise SchemaError(f"{stage.value} must output {expected.__name__}, got {type(output)}")
