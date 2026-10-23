"""Variant ladder, factorial cells, provider overrides and the diagnosis settings."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest
from settings_factory import SettingsFactory

from physverify.core.errors import ConfigError
from physverify.diagnosis.chain_runner import ChainRunner
from physverify.diagnosis.chain_variants import (
    CELLS,
    EXTRACTION_SECTION,
    FACTORS,
    PLAN_KEY,
    apply_overrides,
    cell_name,
    factorial_cells,
    ladder,
    scale_variants,
    truth_covers,
    with_provider,
)
from physverify.diagnosis.settings import diagnosis_settings
from physverify.extraction._base import GROUND_TRUTH
from physverify.schema.clip import Split
from physverify.schema.stages import STAGE_ORDER, Stage

FIXED = frozenset({Stage.DETECT, Stage.CAMERA})


def test_ladder_uses_ground_truth_prefixes() -> None:
    variants = ladder()
    assert [v.name for v in variants] == [f"V{k}" for k in range(len(STAGE_ORDER) + 1)]
    for k, variant in enumerate(variants):
        assert variant.gt_order == STAGE_ORDER[:k]
        assert variant.gt_stages == frozenset(STAGE_ORDER[:k])
        assert dict(variant.overrides) == {}


def test_each_stage_truth_covers_it_and_its_upstream() -> None:
    variants = ladder()
    for k, stage in enumerate(STAGE_ORDER):
        assert truth_covers(variants[k + 1], stage)
        assert not truth_covers(variants[k], stage)
    assert not truth_covers(
        dataclasses.replace(variants[3], gt_stages=frozenset({Stage.DEPTH})), Stage.DEPTH
    )


def test_factorial_has_eight_cells_with_detect_and_camera_fixed() -> None:
    cells = factorial_cells(FIXED)
    assert FACTORS == (Stage.TRACK, Stage.DEPTH, Stage.SCALE)
    assert [c.name for c in cells] == [
        "F_ttt",
        "F_ttp",
        "F_tpt",
        "F_tpp",
        "F_ptt",
        "F_ptp",
        "F_ppt",
        "F_ppp",
    ]
    assert [cell_name(cell) for cell in CELLS] == [c.name for c in cells]
    for cell, variant in zip(CELLS, cells, strict=True):
        assert variant.gt_stages >= FIXED
        perceived = {stage for stage, p in zip(FACTORS, cell, strict=True) if p}
        assert variant.gt_stages == FIXED | (set(FACTORS) - perceived)
    assert cells[0].gt_stages == FIXED | set(FACTORS)
    assert cells[-1].gt_stages == FIXED


def test_scale_variants_keep_the_v5_truth_and_override_scale() -> None:
    variants = scale_variants(("size_prior", "free_fall"))
    assert [v.name for v in variants] == ["V5/scale=size_prior", "V5/scale=free_fall"]
    for variant, name in zip(variants, ("size_prior", "free_fall"), strict=True):
        assert variant.gt_order == STAGE_ORDER[:5] == ladder()[5].gt_order
        assert dict(variant.overrides) == {Stage.SCALE: name}


def test_with_provider_changes_only_that_stage(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    changed = with_provider(settings, Stage.SCALE, "free_fall")
    before = settings.sections[EXTRACTION_SECTION][PLAN_KEY]
    after = changed.sections[EXTRACTION_SECTION][PLAN_KEY]
    assert after[Stage.SCALE.value] == "free_fall"
    assert before[Stage.SCALE.value] == "size_prior"
    assert {k: v for k, v in after.items() if k != Stage.SCALE.value} == {
        k: v for k, v in before.items() if k != Stage.SCALE.value
    }
    extraction = {k: v for k, v in changed.sections[EXTRACTION_SECTION].items() if k != PLAN_KEY}
    assert extraction == {
        k: v for k, v in settings.sections[EXTRACTION_SECTION].items() if k != PLAN_KEY
    }
    assert {k: v for k, v in changed.sections.items() if k != EXTRACTION_SECTION} == {
        k: v for k, v in settings.sections.items() if k != EXTRACTION_SECTION
    }
    for field in dataclasses.fields(settings):
        if field.name != "sections":
            assert getattr(changed, field.name) == getattr(settings, field.name)


def test_apply_overrides_without_overrides_is_identity(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    assert apply_overrides(settings, ladder()[0]) is settings


def test_resolved_plans_follow_the_variants(
    make_settings: SettingsFactory, make_chain: Callable[..., ChainRunner]
) -> None:
    chain = make_chain(make_settings())
    for variant in (*ladder(), *factorial_cells(FIXED), *scale_variants(("free_fall",))):
        plan = chain.plan(variant)
        assert set(plan.ground_truth_stages()) == variant.gt_stages
        for stage in STAGE_ORDER:
            assert (plan.step(stage).name == GROUND_TRUTH) == (stage in variant.gt_stages)
        for stage, name in variant.overrides.items():
            assert plan.names[stage] == name
    assert chain.plan(ladder()[0]).step(Stage.TRACK).name == "diag_fake"


def test_chain_warms_and_reloads_every_stage(
    make_settings: SettingsFactory, make_chain: Callable[..., ChainRunner]
) -> None:
    chain = make_chain(make_settings())
    clips = chain.dataset.clips()
    assert [c.split for c in clips] == [Split.CALIB, Split.EVAL]
    variants = (ladder()[0], ladder()[-1])
    assert chain.warm(clips, variants) == ()
    for clip in clips:
        for variant in variants:
            loaded = chain.outputs(clip, variant)
            assert loaded is not None
            assert set(loaded.outputs) == set(STAGE_ORDER)
            assert loaded.plan.names == chain.plan(variant).names


def test_chain_reports_a_failing_clip(
    make_settings: SettingsFactory, make_chain: Callable[..., ChainRunner]
) -> None:
    settings = make_settings("extraction.providers.track.diag_fake.fail_clip=fixture_fall")
    chain = make_chain(settings)
    clips = chain.dataset.clips()
    failures = chain.warm(clips, (ladder()[0],))
    assert [(f.clip_id, f.stage) for f in failures] == [("fixture_fall", Stage.TRACK)]
    assert chain.outputs(clips[1], ladder()[0]) is None
    assert chain.outputs(clips[0], ladder()[0]) is not None


def test_diagnosis_settings_load_from_the_repo_configs(make_settings: SettingsFactory) -> None:
    diag = diagnosis_settings(make_settings("diagnosis.min_samples=3"))
    assert diag.min_samples == 3
    assert diag.profile_split is Split.CALIB
    assert diag.report_split is Split.EVAL
    assert diag.factorial.fixed_ground_truth == (Stage.DETECT, Stage.CAMERA)
    assert diag.end_to_end.z_levels == (1.0, 2.0)
    assert diag.scale.providers == ("size_prior", "free_fall")


def test_precise_profile_uses_the_full_subset(make_settings: SettingsFactory) -> None:
    assert diagnosis_settings(make_settings(profile="precise")).subset == "full"


@pytest.mark.parametrize(
    "override",
    [
        "diagnosis.metrics.boundary_tol_fracc=0.1",
        'diagnosis.factorial.fixed_ground_truth=["detect","track"]',
        "diagnosis.real.min_fall_frames=2",
    ],
)
def test_invalid_diagnosis_settings_fail_loudly(
    make_settings: SettingsFactory, override: str
) -> None:
    with pytest.raises(ConfigError):
        diagnosis_settings(make_settings(override))
