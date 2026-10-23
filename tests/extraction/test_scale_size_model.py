"""ADR 0010 I2-I3: the learned size scale, its equivalence with the frozen prior and its key."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
from support import SettingsFactory, chain_keys, make_context, run_provider
from synthetic_clip import SyntheticClip
from test_scale import PRIORS, truth_lift
from wiring import resolve_plan

from physverify.config.models import SizePriorTable
from physverify.extraction._base import StageProvider
from physverify.extraction.providers.scale_size_model import SizeModelScale
from physverify.extraction.providers.scale_size_prior import SizePriorScale
from physverify.physics.geometry import characteristic_lengths
from physverify.schema.scale import ScaleEstimate, ScaleSource
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable
from physverify.sizing.model import ClassSize, SizeModelTable
from physverify.sizing.settings import FixedSize, SizeLookup

SCALE = 3.7
CUP = ClassSize(mu_log=math.log(0.09), sigma_log=0.15, n_objects=100, n_videos=90)
FULL = SizeModelTable(source="test", sigma_factor=0.9, classes={"cup": CUP, "shoe": CUP})


def run_model(clip: SyntheticClip, lift: ObjectStateTable, **params: object) -> ScaleEstimate:
    ctx = make_context(clip, upstream={Stage.LIFT: lift})
    output = run_provider(SizeModelScale, ctx, **params)
    assert isinstance(output, ScaleEstimate)
    return output


def run_prior(clip: SyntheticClip, lift: ObjectStateTable) -> ScaleEstimate:
    ctx = make_context(clip, upstream={Stage.LIFT: lift})
    output = run_provider(SizePriorScale, ctx, priors=SizePriorTable(PRIORS))
    assert isinstance(output, ScaleEstimate)
    return output


def with_size_sigma(lift: ObjectStateTable, rel: float) -> ObjectStateTable:
    return replace(lift, size_sigma=lift.size * rel)


def object_length(lift: ObjectStateTable, n: int) -> float:
    size = lift.size[:, n]
    known = lift.visible[:, n] & np.all(np.isfinite(size), axis=1) & np.all(size > 0, axis=1)
    return float(np.median(characteristic_lengths(size[known])))


@pytest.mark.parametrize("model", [SizeModelTable(), FULL], ids=["empty", "full"])
def test_kubric_labels_give_exactly_the_frozen_prior_scale(
    fixture_clip: SyntheticClip, model: SizeModelTable
) -> None:
    lift = with_size_sigma(truth_lift(fixture_clip, scale=SCALE), 0.05)
    expected = run_prior(fixture_clip, lift)
    got = run_model(
        fixture_clip, lift, priors=SizePriorTable(PRIORS), model=model, same_class_rho=0.0
    )
    assert expected.meters_per_unit is not None and got.meters_per_unit is not None
    assert abs(got.meters_per_unit / expected.meters_per_unit - 1) < 1e-12
    assert got.sigma == expected.sigma
    assert got.source is ScaleSource.SIZE_PRIOR
    assert got.detail.startswith("size_model:")


def test_one_class_object_follows_the_log_normal_formula(fixture_clip: SyntheticClip) -> None:
    lift = with_size_sigma(truth_lift(fixture_clip, scale=SCALE), 0.06)
    lookup = SizeLookup(labels={"ball": "cup"})
    got = run_model(fixture_clip, lift, model=FULL, lookup=lookup)
    n = lift.labels.tolist().index("ball")
    measured = math.sqrt(3 * 0.06**2) / 3
    assert got.meters_per_unit is not None
    assert abs(got.meters_per_unit / (0.09 / object_length(lift, n)) - 1) < 1e-12
    expected_rel = math.sqrt((0.9 * 0.15) ** 2 + measured**2)
    assert abs(got.sigma / got.meters_per_unit - expected_rel) < 1e-12


def test_a_fixed_size_and_a_synonym_resolve(fixture_clip: SyntheticClip) -> None:
    lift = truth_lift(fixture_clip, scale=SCALE)
    fixed = {"tennis ball": FixedSize(mean_m=0.067, rel_spread=0.05, source="ITF")}
    lookup = SizeLookup(labels={"ball": "tennis ball"}, fixed=fixed)
    got = run_model(fixture_clip, lift, lookup=lookup)
    n = lift.labels.tolist().index("ball")
    assert got.meters_per_unit is not None
    assert abs(got.meters_per_unit / (0.067 / object_length(lift, n)) - 1) < 1e-12
    assert abs(got.sigma / got.meters_per_unit - 0.05) < 1e-12


@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_two_objects_of_one_class_share_the_prior_term(
    fixture_clip: SyntheticClip, rho: float
) -> None:
    lift = with_size_sigma(truth_lift(fixture_clip, scale=SCALE), 0.06)
    lookup = SizeLookup(labels={"ball": "cup", "box": "cup"})
    got = run_model(fixture_clip, lift, model=FULL, lookup=lookup, same_class_rho=rho)
    p2 = (0.9 * 0.15) ** 2
    q2 = 3 * 0.06**2 / 9
    logs = np.array([math.log(0.09 / object_length(lift, n)) for n in range(2)])
    # Compound symmetry, n = 2 equal-variance members: rho p^2 + ((1 - rho) p^2 + q^2) / n.
    expected_rel_var = rho * p2 + ((1 - rho) * p2 + q2) / 2
    assert got.meters_per_unit is not None
    assert abs(math.log(got.meters_per_unit) - logs.mean()) < 1e-12
    assert abs((got.sigma / got.meters_per_unit) ** 2 - expected_rel_var) < 1e-12


def test_no_label_with_a_size_gives_no_scale(fixture_clip: SyntheticClip) -> None:
    got = run_model(fixture_clip, truth_lift(fixture_clip, scale=SCALE))
    assert got.source is ScaleSource.NONE and got.meters_per_unit is None


def changed_stages(
    make_settings: SettingsFactory,
    providers: Mapping[str, type[StageProvider]],
    clip: SyntheticClip,
    *overrides: str,
) -> list[Stage]:
    base = ("extraction.plan.scale=size_model",)
    plans = [
        resolve_plan(make_settings(*base, *extra), providers, dataset="x")
        for extra in ((), overrides)
    ]
    keys = [chain_keys(p, clip.manifest, gt_version=None, profile=None) for p in plans]
    return [stage for stage in STAGE_ORDER if keys[0][stage] != keys[1][stage]]


def test_a_model_edit_invalidates_only_the_scale(
    make_settings: SettingsFactory,
    real_providers: Mapping[str, type[StageProvider]],
    fixture_clip: SyntheticClip,
) -> None:
    args = (make_settings, real_providers, fixture_clip)
    assert changed_stages(*args, "size_model.sigma_factor=1.5") == [Stage.SCALE]
    assert changed_stages(*args, 'sizing.labels.teacup="cup"') == [Stage.SCALE]
    assert changed_stages(*args, "sizing.workers=2", "sizing.per_class_test=7") == []
