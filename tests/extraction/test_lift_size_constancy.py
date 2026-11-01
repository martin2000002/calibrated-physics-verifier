"""ADR 0033: ``lift/backproject_size_constancy`` is ``lift/backproject`` with the along-ray
sigma of sphere tracks swapped; every other field is the parent's."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from support import entry, make_context, profile_of, run_provider
from synthetic_clip import SyntheticClip

from physverify.extraction._base import GROUND_TRUTH, StageProvider
from physverify.extraction.lift_params import SizeConstancyLiftParams, size_constancy_params
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.extraction.providers.lift_size_constancy import SizeConstancyLift
from physverify.extraction.size_constancy import along_ray_sigma, replace_along_ray_sigma
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import MaskTrack
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable, PrimitiveKind

FIXTURE_PARAMS = {"min_diameter_px": 4.0}
FIELDS = ("position", "size", "size_sigma", "quaternion", "visible", "in_frame", "border_cut")


def truth_inputs(clip: SyntheticClip) -> dict[Stage, StageOutput]:
    s = make_context(clip).sampling
    gt = clip.ground_truth
    return {Stage.TRACK: gt.masks(s), Stage.DEPTH: gt.depth(s), Stage.CAMERA: gt.camera(s)}


def lifted(
    clip: SyntheticClip, provider: type[StageProvider], **params: object
) -> tuple[ObjectStateTable, MaskTrack]:
    upstream: Mapping[Stage, StageOutput] = truth_inputs(clip)
    names = {**dict.fromkeys(STAGE_ORDER, GROUND_TRUTH), Stage.LIFT: provider.name}
    profile = profile_of(
        entry(Stage.LIFT, provider.name, "centroid_rel", 0.1),
        entry(Stage.LIFT, provider.name, "size_rel", 0.05),
    )
    ctx = make_context(clip, upstream=upstream, plan=names, profile=profile)
    table = run_provider(provider, ctx, **params)
    track = upstream[Stage.TRACK]
    assert isinstance(table, ObjectStateTable)
    assert isinstance(track, MaskTrack)
    return table, track


@pytest.fixture
def pair(fixture_clip: SyntheticClip) -> tuple[ObjectStateTable, ObjectStateTable, MaskTrack]:
    parent, _ = lifted(fixture_clip, BackprojectLift)
    table, track = lifted(fixture_clip, SizeConstancyLift, **FIXTURE_PARAMS)
    return parent, table, track


def test_every_field_but_position_sigma_is_the_parents(
    pair: tuple[ObjectStateTable, ObjectStateTable, MaskTrack],
) -> None:
    parent, table, _ = pair
    for field in FIELDS:
        np.testing.assert_array_equal(getattr(table, field), getattr(parent, field), err_msg=field)
    np.testing.assert_array_equal(table.primitive, parent.primitive)


def test_position_sigma_is_the_parents_with_the_along_ray_part_swapped(
    pair: tuple[ObjectStateTable, ObjectStateTable, MaskTrack],
) -> None:
    parent, table, track = pair
    spheres = np.asarray(parent.primitive) == PrimitiveKind.SPHERE.value
    params = size_constancy_params(SizeConstancyLiftParams(**FIXTURE_PARAMS))
    along = along_ray_sigma(
        np.asarray(track.masks, np.bool_), parent.position[..., 2], parent.visible, spheres, params
    )
    expected = replace_along_ray_sigma(parent.position, parent.position_sigma, along)
    np.testing.assert_array_equal(table.position_sigma, expected)
    assert bool(spheres.any()) and bool(np.isfinite(along).any())
    changed = (table.position_sigma != parent.position_sigma).any(axis=-1)
    assert bool(changed.any())
    assert not bool(changed[:, ~spheres].any())


def test_with_the_default_minimum_the_small_fixture_sphere_keeps_the_parents_sigma(
    fixture_clip: SyntheticClip,
) -> None:
    parent, _ = lifted(fixture_clip, BackprojectLift)
    table, _ = lifted(fixture_clip, SizeConstancyLift)
    np.testing.assert_array_equal(table.position_sigma, parent.position_sigma)
