"""Ground-truth contacts in TRACK ids and LIFT units; peak_force stays in newtons."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from physverify.extraction._base import GROUND_TRUTH, StageContext, WeightSpec
from physverify.extraction.matching import remap_contacts
from physverify.extraction.providers._common import ProviderParams, expect_params
from physverify.extraction.providers._truth import conditional_truth
from physverify.schema.contacts import ContactTable
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame


class Params(ProviderParams):
    pass


class GroundTruthContacts:
    stage: ClassVar[Stage] = Stage.CONTACTS
    name: ClassVar[str] = GROUND_TRUTH
    version: ClassVar[str] = "2"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.TRACK, Stage.LIFT)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = True
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ContactTable:
        truth = conditional_truth(ctx)
        frames = ctx.sampling.frame_indices
        return remap_contacts(
            ctx.require_gt().contacts(CoordFrame.CAMERA_OPENCV),
            truth.aligned.ids,
            source_key=truth.lift.source_key,
            unit=truth.lift.unit,
            length_scale=truth.scale(),
            window=(float(frames[0]), float(frames[-1])),
        )


PLUGINS = (GroundTruthContacts,)
