"""Scale from objects of a regulation size (``[sizing.fixed]``): s = D / median L per object,
combined and focal-corrected as ``size_prior`` does. No object of a fixed size: no scale."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import ClassVar

from pydantic import BaseModel

from physverify.core.models import FrozenModel
from physverify.extraction._base import StageContext, WeightSpec
from physverify.extraction.providers._common import expect_params
from physverify.extraction.providers._size_prior import (
    clip_focal_phi,
    combine,
    object_scale,
    with_focal,
)
from physverify.physics.length_sigma import DEFAULT_AXIS_ERRORS, AxisErrors
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable
from physverify.sizing.model import SizeModelTable, predict
from physverify.sizing.settings import SECTION as SIZING_SECTION
from physverify.sizing.settings import SizeLookup

NO_ANCHOR = "no_known_size"
# No learned classes: only the curated regulation sizes (and their synonyms) are anchors.
FIXED_ONLY = SizeModelTable()


class Params(FrozenModel):
    lookup: SizeLookup = SizeLookup()
    axis_errors: AxisErrors = DEFAULT_AXIS_ERRORS


class KnownSizeScale:
    """A regulation size is a size prior with a small spread, so the source is ``size_prior``;
    the provider key in the provenance tells the two apart."""

    stage: ClassVar[Stage] = Stage.SCALE
    name: ClassVar[str] = "known_size"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.CAMERA, Stage.LIFT)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[WeightSpec, ...]] = ()
    Params: ClassVar[type[BaseModel]] = Params
    settings_sections: ClassVar[Mapping[str, tuple[str, type[BaseModel]]]] = {
        "lookup": (SIZING_SECTION, SizeLookup)
    }

    def __init__(self, params: BaseModel) -> None:
        self.params = expect_params(params, Params)

    def run(self, ctx: StageContext) -> ScaleEstimate:
        lift = ctx.require(Stage.LIFT, ObjectStateTable)
        phi = clip_focal_phi(ctx)
        estimates = [
            estimate
            for n, label in enumerate(lift.labels.tolist())
            if (size := predict(label, FIXED_ONLY, self.params.lookup)) is not None
            and (estimate := object_scale(lift, n, size, phi, self.params.axis_errors)) is not None
        ]
        if not estimates:
            return ScaleEstimate(None, math.nan, ScaleSource.NONE, lift.source_key, NO_ANCHOR)
        scale, sigma = combine(estimates)
        return ScaleEstimate(
            scale, with_focal(scale, sigma, phi), ScaleSource.SIZE_PRIOR, lift.source_key
        )


PLUGINS = (KnownSizeScale,)
