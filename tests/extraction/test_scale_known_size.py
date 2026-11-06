"""ADR 0053: scale from a regulation-size anchor; no anchor, no scale."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from support import make_context, run_provider
from synthetic_clip import SyntheticClip
from test_scale import LIFT_KEY, size_prior, truth_camera, truth_lift

from physverify.config.models import SizePrior
from physverify.extraction.providers.scale_known_size import NO_ANCHOR, KnownSizeScale
from physverify.schema.scale import ScaleEstimate, ScaleSource, effective_scale
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable
from physverify.sizing.settings import FixedSize, SizeLookup

FIXED = {"basketball": FixedSize(mean_m=0.243, rel_spread=0.05, source="FIBA")}
SIZES = {"basketball": SizePrior(mean_m=0.243, rel_spread=0.05)}


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
