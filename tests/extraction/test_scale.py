"""AC7: metric scale from size priors and from free fall."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from support import make_context, run_provider, sampling_of
from synthetic_clip import SyntheticClip

from physverify.config.models import SizePrior, SizePriorTable
from physverify.extraction.matching import align_ground_truth
from physverify.extraction.providers.scale_free_fall import FreeFallScale
from physverify.extraction.providers.scale_size_prior import (
    ObjectScale,
    SizePriorScale,
    combine,
)
from physverify.physics.geometry import characteristic_lengths
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import Stage
from physverify.schema.states import LengthUnit, ObjectStateTable

LIFT_KEY = "lift-key"
SCALE = 7.3
PRIORS = {
    "ball": SizePrior(mean_m=0.3, rel_spread=0.2),
    "cube": SizePrior(mean_m=0.4, rel_spread=0.3),
}


def truth_lift(clip: SyntheticClip, *, scale: float = 1.0) -> ObjectStateTable:
    sampling = sampling_of(clip.manifest)
    truth = clip.ground_truth
    states = align_ground_truth(truth, truth.masks(sampling), sampling).states
    return replace(
        states,
        unit=LengthUnit.METER if scale == 1.0 else LengthUnit.RELATIVE,
        source_key=LIFT_KEY,
        position=states.position / scale,
        size=states.size / scale,
    )


def size_prior(
    clip: SyntheticClip, lift: ObjectStateTable, priors: dict[str, SizePrior]
) -> ScaleEstimate:
    ctx = make_context(clip, upstream={Stage.LIFT: lift})
    output = run_provider(SizePriorScale, ctx, priors=SizePriorTable(priors))
    assert isinstance(output, ScaleEstimate)
    return output


def free_fall(clip: SyntheticClip, lift: ObjectStateTable) -> ScaleEstimate:
    output = run_provider(FreeFallScale, make_context(clip, upstream={Stage.LIFT: lift}))
    assert isinstance(output, ScaleEstimate)
    return output


def only(lift: ObjectStateTable, object_id: str) -> ObjectStateTable:
    n = lift.object_index(object_id)
    per_object = {
        name: getattr(lift, name)[:, [n]]
        for name, spec in lift._arrays.items()
        if spec.shape[:2] == ("T", "N")
    }
    return replace(
        lift,
        object_ids=lift.object_ids[[n]],
        labels=lift.labels[[n]],
        primitive=lift.primitive[[n]],
        **per_object,
    )


def test_size_prior_follows_the_log_inverse_variance_formula(
    fixture_clip: SyntheticClip,
) -> None:
    lift = truth_lift(fixture_clip, scale=SCALE)
    lift = replace(lift, size_sigma=0.05 * lift.size)
    estimate = size_prior(fixture_clip, lift, PRIORS)
    logs, weights = [], []
    for label, prior in (("ball", PRIORS["ball"]), ("box", PRIORS["cube"])):
        n = lift.object_index(label)
        length = float(np.median(characteristic_lengths(lift.size[lift.visible[:, n], n])))
        rel_var = prior.rel_spread**2 + (math.sqrt(3 * 0.05**2) / 3.0) ** 2
        logs.append(math.log(prior.mean_m / length))
        weights.append(1.0 / rel_var)
    expected = math.exp(float(np.dot(weights, logs) / np.sum(weights)))
    assert estimate.source is ScaleSource.SIZE_PRIOR
    assert estimate.meters_per_unit == pytest.approx(expected, abs=1e-12)
    assert estimate.sigma == pytest.approx(expected / math.sqrt(sum(weights)), abs=1e-12)
    assert estimate.lift_key == lift.source_key == LIFT_KEY


def test_same_prior_objects_shrink_sigma_by_root_n(fixture_clip: SyntheticClip) -> None:
    lift = truth_lift(fixture_clip, scale=SCALE)
    twins = replace(lift, labels=np.array(["ball", "ball"]), size_sigma=np.zeros_like(lift.size))
    estimate = size_prior(fixture_clip, twins, PRIORS)
    assert estimate.meters_per_unit is not None
    rel = estimate.sigma / estimate.meters_per_unit
    assert rel == pytest.approx(PRIORS["ball"].rel_spread / math.sqrt(2), abs=1e-12)


def test_labels_without_prior_give_no_scale(fixture_clip: SyntheticClip) -> None:
    lift = replace(truth_lift(fixture_clip), labels=np.array(["widget", "gizmo"]))
    estimate = size_prior(fixture_clip, lift, PRIORS)
    assert estimate.source is ScaleSource.NONE
    assert estimate.meters_per_unit is None
    assert estimate.lift_key == LIFT_KEY


def test_free_fall_recovers_unit_scale_from_the_falling_sphere(
    fixture_clip: SyntheticClip,
) -> None:
    estimate = free_fall(fixture_clip, truth_lift(fixture_clip))
    assert estimate.source is ScaleSource.FREE_FALL
    assert estimate.meters_per_unit == pytest.approx(1.0, abs=1e-6)
    assert math.isfinite(estimate.sigma)
    assert estimate.lift_key == LIFT_KEY
    relative = free_fall(fixture_clip, truth_lift(fixture_clip, scale=SCALE))
    assert relative.meters_per_unit == pytest.approx(SCALE, rel=1e-6)


def test_free_fall_rejects_the_resting_box(fixture_clip: SyntheticClip) -> None:
    estimate = free_fall(fixture_clip, only(truth_lift(fixture_clip), "box"))
    assert estimate.source is ScaleSource.NONE
    assert estimate.meters_per_unit is None


def test_exact_objects_that_disagree_report_their_spread() -> None:
    logs = [math.log(2.0), math.log(2.4), math.log(2.2)]
    scale, sigma = combine([ObjectScale(log, 0.0) for log in logs] + [ObjectScale(0.0, 0.5)])
    assert scale == pytest.approx(math.exp(float(np.mean(logs))), abs=1e-12)
    expected = scale * float(np.std(logs, ddof=1)) / math.sqrt(len(logs))
    assert sigma == pytest.approx(expected, abs=1e-12)
    assert combine([ObjectScale(logs[0], 0.0), ObjectScale(logs[0], 0.0)]) == (
        pytest.approx(2.0, abs=1e-12),
        0.0,
    )
    assert combine([ObjectScale(logs[0], 0.0)])[1] == 0.0
