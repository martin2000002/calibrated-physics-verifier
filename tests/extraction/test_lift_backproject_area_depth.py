"""``lift/backproject_area_depth`` is ``lift/backproject`` rescaled by the area depth, on a
perceived TRACK and on a true one alike."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from support import entry, make_context, profile_of, run_provider
from synthetic_clip import SyntheticClip

from physverify.app.plugins import PLUGIN_KINDS
from physverify.extraction._base import GROUND_TRUTH, StageProvider
from physverify.extraction.area_depth import (
    SCALED_FIELDS,
    area_depth_factor,
    area_depth_rows,
    mask_area,
    scale_depth_fields,
)
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.extraction.providers.lift_backproject_area_depth import (
    AreaDepthLift,
    AreaDepthLiftParams,
)
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable

PERCEIVED_TRACK = "perceived_track"
UNSCALED = ("quaternion", "visible", "in_frame", "border_cut", "nearer_cover", "primitive")


def lifted(clip: SyntheticClip, provider: type[StageProvider], track: str) -> ObjectStateTable:
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
        entry(Stage.LIFT, provider.name, "centroid_rel", 0.1),
        entry(Stage.LIFT, provider.name, "size_rel", 0.05),
    )
    ctx = make_context(clip, upstream=upstream, plan=names, profile=profile)
    table = run_provider(provider, ctx)
    assert isinstance(table, ObjectStateTable)
    return table


def masks_of(clip: SyntheticClip) -> MaskTrack:
    track = clip.ground_truth.masks(make_context(clip).sampling)
    assert isinstance(track, MaskTrack)
    return track


def test_the_provider_is_registered_under_lift() -> None:
    assert PLUGIN_KINDS["providers"]()["lift/backproject_area_depth"] is AreaDepthLift
    assert AreaDepthLift.Params is AreaDepthLiftParams


@pytest.mark.parametrize("track", [PERCEIVED_TRACK, GROUND_TRUTH])
def test_the_parent_is_rescaled_by_the_area_depth(fixture_clip: SyntheticClip, track: str) -> None:
    parent = lifted(fixture_clip, BackprojectLift, track)
    table = lifted(fixture_clip, AreaDepthLift, track)
    area = mask_area(np.asarray(masks_of(fixture_clip).masks, dtype=np.bool_))
    used = area_depth_rows(parent, area)
    factor = area_depth_factor(parent.position[..., 2], area, used, AreaDepthLiftParams().min_rows)
    expected = scale_depth_fields(parent, factor)
    assert bool((factor != 1.0).any())
    for name in SCALED_FIELDS:
        np.testing.assert_array_equal(getattr(table, name), getattr(expected, name), err_msg=name)
    for name in UNSCALED:
        np.testing.assert_array_equal(getattr(table, name), getattr(parent, name), err_msg=name)
