"""``lift/backproject_contour_circle`` is ``lift/backproject`` with ``centre = "contour_circle"``,
field for field, when each reads the same error-profile values under its own name."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np
import pytest
from support import entry, make_context, profile_of, run_provider
from synthetic_clip import SyntheticClip

from physverify.app.plugins import PLUGIN_KINDS
from physverify.extraction._base import GROUND_TRUTH, StageProvider, derives_depth_from_size
from physverify.extraction.centres._base import CentreMethod
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.extraction.providers.lift_backproject_contour_circle import (
    ContourCircleLift,
    ContourCircleLiftParams,
)
from physverify.schema.error_profile import CENTROID_REL, ORIENTATION_RAD, SIZE_REL
from physverify.schema.outputs import StageOutput
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable

CONTOUR = "contour_circle"
PERCEIVED_TRACK = "perceived_track"


def lifted(
    clip: SyntheticClip, provider: type[StageProvider], track: str, **params: str
) -> ObjectStateTable:
    s = make_context(clip).sampling
    gt = clip.ground_truth
    upstream: Mapping[Stage, StageOutput] = {
        Stage.TRACK: gt.masks(s),
        Stage.DEPTH: gt.depth(s),
        Stage.CAMERA: gt.camera(s),
    }
    names = {**dict.fromkeys(STAGE_ORDER, GROUND_TRUTH), Stage.TRACK: track}
    names[Stage.LIFT] = provider.name
    profile = profile_of(
        entry(Stage.LIFT, provider.name, CENTROID_REL, 0.1),
        entry(Stage.LIFT, provider.name, SIZE_REL, 0.05),
        entry(Stage.LIFT, provider.name, ORIENTATION_RAD, 0.2),
    )
    ctx = make_context(clip, upstream=upstream, plan=names, profile=profile)
    table = run_provider(provider, ctx, **params)
    assert isinstance(table, ObjectStateTable)
    return table


def test_the_provider_is_registered_under_lift() -> None:
    assert PLUGIN_KINDS["providers"]()["lift/backproject_contour_circle"] is ContourCircleLift
    assert ContourCircleLift.Params is ContourCircleLiftParams
    assert not derives_depth_from_size(ContourCircleLift)


def test_the_centre_always_enters_the_params_dump() -> None:
    assert ContourCircleLiftParams().model_dump()["centre"] == CONTOUR
    assert "centre" not in BackprojectLift.Params().model_dump()


@pytest.mark.parametrize("track", [PERCEIVED_TRACK, GROUND_TRUTH])
def test_every_field_equals_the_main_lift_with_the_contour_circle_centre(
    fixture_clip: SyntheticClip, track: str
) -> None:
    parent = lifted(fixture_clip, BackprojectLift, track, centre=CONTOUR)
    table = lifted(fixture_clip, ContourCircleLift, track)
    assert parent.centre_method is not None
    assert bool((parent.centre_method != CentreMethod.SURFACE_MEDIAN.value).any())
    assert bool(np.isfinite(parent.position_sigma).any())
    for field in dataclasses.fields(ObjectStateTable):
        mine, theirs = getattr(table, field.name), getattr(parent, field.name)
        if isinstance(theirs, np.ndarray):
            np.testing.assert_array_equal(mine, theirs, err_msg=field.name, strict=True)
        else:
            assert mine == theirs, field.name
