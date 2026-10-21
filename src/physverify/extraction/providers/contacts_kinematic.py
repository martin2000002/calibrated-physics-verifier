"""Contacts with kinematic impacts (ADR 0005 4.9, it3 amendment).

Proximity contacts (``_proximity.py``) plus impacts found as velocity kinks of each track
(``_impacts.py``) and attributed to a surface (``_attribution.py``), so that an impact between
two sampled rows is not lost. Resting intervals are split only when the normal turns
(``resting_split = "normal"``); with ``impacts = "none"`` and ``resting_split =
"normal_and_count"`` the tables are those of ``contacts/proximity``.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from physverify.extraction._base import StageContext, WeightSpec
from physverify.extraction.providers._attribution import (
    KinematicScene,
    camera_positions,
    kinematic_records,
)
from physverify.extraction.providers._common import expect_params
from physverify.extraction.providers._proximity import ContactScene, ProximityParams
from physverify.schema.contacts import ContactRecord, ContactTable
from physverify.schema.perception import CameraModel
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable


class Params(ProximityParams):
    impacts: Literal["kinematic", "none"] = "kinematic"
    resting_split: Literal["normal", "normal_and_count"] = "normal"
    side_rows: int = Field(default=4, ge=3)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)


class KinematicContacts:
    stage: ClassVar[Stage] = Stage.CONTACTS
    name: ClassVar[str] = "kinematic"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.CAMERA, Stage.KINEMATICS)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ContactTable:
        states = ctx.require(Stage.KINEMATICS, ObjectStateTable)
        camera = ctx.require(Stage.CAMERA, CameraModel)
        return ContactTable(
            clip_id=states.clip_id,
            frame=states.frame,
            unit=states.unit,
            source_key=states.source_key,
            records=self.records(states, camera, ctx.sampling.seconds_per_frame),
        )

    def records(
        self, states: ObjectStateTable, camera: CameraModel, seconds_per_frame: float
    ) -> tuple[ContactRecord, ...]:
        p = self.params
        split = p.resting_split == "normal_and_count"
        proximity = ContactScene(states, p, split_on_point_count=split)
        found = tuple(proximity.records())
        if p.impacts == "none":
            return found
        scene = KinematicScene(
            states=states,
            proximity=proximity,
            records=found,
            camera_positions=camera_positions(states, camera),
            seconds_per_frame=seconds_per_frame,
            tol_rel=p.tol_rel,
        )
        return kinematic_records(scene, float(camera.K[0, 0]), p.side_rows, p.alpha)


PLUGINS = (KinematicContacts,)
