"""Contacts from the proximity of primitives fitted to the kinematic states."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from physverify.extraction._base import StageContext, WeightSpec
from physverify.extraction.providers._common import expect_params
from physverify.extraction.providers._proximity import ContactScene
from physverify.extraction.providers._proximity import ProximityParams as Params
from physverify.schema.contacts import ContactRecord, ContactTable
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable


class ProximityContacts:
    stage: ClassVar[Stage] = Stage.CONTACTS
    name: ClassVar[str] = "proximity"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.KINEMATICS,)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ContactTable:
        states = ctx.require(Stage.KINEMATICS, ObjectStateTable)
        return ContactTable(
            clip_id=states.clip_id,
            frame=states.frame,
            unit=states.unit,
            source_key=states.source_key,
            records=self.records(states, ctx.sampling.seconds_per_frame),
        )

    def records(
        self, states: ObjectStateTable, seconds_per_frame: float
    ) -> tuple[ContactRecord, ...]:
        scene = ContactScene(states, self.params, seconds_per_frame, split_on_point_count=True)
        return tuple(scene.records())


PLUGINS = (ProximityContacts,)
