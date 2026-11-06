"""Scale from a regulation-size anchor; no anchor, no scale."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from support import entry, make_context, profile_of, run_provider
from synthetic_clip import SyntheticClip
from test_scale import LIFT_KEY, size_prior, truth_camera, truth_lift

from physverify.config.models import SizePrior
from physverify.extraction._base import GROUND_TRUTH
from physverify.extraction.providers.lift_backproject_area_depth import AreaDepthLift
from physverify.extraction.providers.scale_known_size import NO_ANCHOR, KnownSizeScale
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.scale import ScaleEstimate, ScaleSource, effective_scale
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable
from physverify.sizing.settings import FixedSize, SizeLookup

FIXED = {"basketball": FixedSize(mean_m=0.243, rel_spread=0.05, source="FIBA")}
SIZES = {"basketball": SizePrior(mean_m=0.243, rel_spread=0.05)}
SIZE_REL = 0.1


def known_size(clip: SyntheticClip, lift: ObjectStateTable) -> ScaleEstimate:
    ctx = make_context(clip, upstream={Stage.LIFT: lift, Stage.CAMERA: truth_camera(clip)})
    output = run_provider(KnownSizeScale, ctx, lookup=SizeLookup(fixed=FIXED))
    assert isinstance(output, ScaleEstimate)
    return output


def relabelled(clip: SyntheticClip, label: str) -> ObjectStateTable:
    lift = truth_lift(clip, scale=4.0)
    return replace(lift, labels=np.array([label] * len(lift.labels)))


def test_an_anchor_gives_the_size_prior_scale_of_its_regulation_size(
    fixture_clip: SyntheticClip,
) -> None:
    lift = relabelled(fixture_clip, "basketball")
    estimate = known_size(fixture_clip, lift)
    reference = size_prior(fixture_clip, lift, SIZES)
    assert estimate.source is ScaleSource.SIZE_PRIOR
    assert estimate.lift_key == LIFT_KEY
    assert estimate.meters_per_unit == pytest.approx(reference.meters_per_unit, rel=1e-12)
    assert estimate.sigma == pytest.approx(reference.sigma, rel=1e-12)


@pytest.mark.parametrize("label", ["ball", "tennis ball", "widget"])
def test_no_anchor_gives_no_scale(fixture_clip: SyntheticClip, label: str) -> None:
    """No scale reaches R5b, so it abstains: SCALE_UNAVAILABLE (tests/checks/test_abstentions)."""
    lift = relabelled(fixture_clip, label)
    estimate = known_size(fixture_clip, lift)
    assert estimate.source is ScaleSource.NONE
    assert estimate.meters_per_unit is None
    assert estimate.detail == NO_ANCHOR
    assert effective_scale(lift, estimate) is None


def area_depth_anchor(clip: SyntheticClip, profile: ErrorProfile | None) -> ObjectStateTable:
    """The anchor route's lift on true track, depth and camera, every track a basketball."""
    sampling = make_context(clip).sampling
    gt = clip.ground_truth
    upstream = {
        Stage.TRACK: gt.masks(sampling),
        Stage.DEPTH: gt.depth(sampling),
        Stage.CAMERA: gt.camera(sampling),
    }
    plan = {**dict.fromkeys(STAGE_ORDER, GROUND_TRUTH), Stage.LIFT: AreaDepthLift.name}
    ctx = make_context(clip, upstream=upstream, plan=plan, profile=profile)
    table = run_provider(AreaDepthLift, ctx)
    assert isinstance(table, ObjectStateTable)
    return replace(table, labels=np.array(["basketball"] * len(table.labels)))


def test_the_lift_profile_gives_the_anchor_a_finite_sigma(fixture_clip: SyntheticClip) -> None:
    """True upstream and camera add no error, so rho^2 = rel_spread^2 + size_rel^2 per object."""
    profile = profile_of(
        entry(Stage.LIFT, AreaDepthLift.name, "centroid_rel", 0.08),
        entry(Stage.LIFT, AreaDepthLift.name, "size_rel", SIZE_REL),
    )
    lift = area_depth_anchor(fixture_clip, profile)
    estimate = known_size(fixture_clip, lift)
    sized = lift.visible & np.all(np.isfinite(lift.size), axis=2) & np.all(lift.size > 0, axis=2)
    objects = int(np.count_nonzero(sized.any(axis=0)))
    assert estimate.meters_per_unit is not None and objects > 0
    rel_sigma = estimate.sigma / estimate.meters_per_unit
    assert rel_sigma == pytest.approx(math.sqrt((0.05**2 + SIZE_REL**2) / objects), rel=1e-9)


def test_without_the_lift_profile_the_anchor_sigma_is_unknown(fixture_clip: SyntheticClip) -> None:
    estimate = known_size(fixture_clip, area_depth_anchor(fixture_clip, None))
    assert estimate.meters_per_unit is not None
    assert math.isnan(estimate.sigma)
