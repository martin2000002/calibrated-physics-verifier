"""AC11: the active profile resolved into one provider and validated parameters per stage."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel
from support import SettingsFactory
from wiring import resolve_plan

from physverify.config.models import SizePriorTable
from physverify.core.errors import ConfigError
from physverify.extraction._base import GROUND_TRUTH, StageProvider
from physverify.extraction.providers._common import ProviderParams
from physverify.schema.stages import STAGE_ORDER, Stage

Providers = Mapping[str, type[StageProvider]]


class BackwardsLift:
    stage: ClassVar[Stage] = Stage.LIFT
    name: ClassVar[str] = "backwards"
    version: ClassVar[str] = "1"
    requires: ClassVar[tuple[Stage, ...]] = (Stage.SCALE,)
    heavy: ClassVar[bool] = False
    is_ground_truth: ClassVar[bool] = False
    error_profile_reads: ClassVar[tuple[Stage, ...]] = ()
    weights: ClassVar[tuple[Any, ...]] = ()
    Params: ClassVar[type[BaseModel]] = ProviderParams


def test_fast_and_precise_come_from_the_extraction_config(
    make_settings: SettingsFactory, real_providers: Providers
) -> None:
    fast = resolve_plan(make_settings(), real_providers, dataset="kubric_synthetic")
    precise = resolve_plan(
        make_settings(profile="precise"), real_providers, dataset="kubric_synthetic"
    )
    assert fast.names[Stage.TRACK] == "edgetam" and fast.sampling.max_side == 640
    assert precise.names[Stage.TRACK] == "sam2_1_hiera_tiny" and precise.sampling.max_side == 960
    assert fast.names[Stage.DETECT] == "grounding_dino_tiny"
    assert precise.names[Stage.DETECT] == "grounding_dino_base"
    assert [step.stage for step in fast.stages] == list(STAGE_ORDER)
    assert fast.names[Stage.LIFT] == precise.names[Stage.LIFT] == "backproject"
    assert fast.ground_truth_stages() == ()


def test_gt_substitutes_the_named_stages(
    make_settings: SettingsFactory, real_providers: Providers
) -> None:
    plan = resolve_plan(
        make_settings(), real_providers, dataset="x", gt_stages=(Stage.DEPTH, Stage.SCALE)
    )
    assert plan.ground_truth_stages() == (Stage.DEPTH, Stage.SCALE)
    assert plan.names[Stage.DEPTH] == plan.names[Stage.SCALE] == GROUND_TRUTH
    assert plan.names[Stage.TRACK] == "edgetam"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("extraction.plan.lift=nope", "unknown provider 'lift/nope'"),
        ("extraction.providers.kinematics.savgol.window=4", "savgol] is invalid"),
        ("extraction.providers.lift.backproject.unknown_field=1", "backproject] is invalid"),
        (
            "extraction.providers.scale.size_prior.priors={ball={mean_m=1.0,rel_spread=0.1}}",
            "comes from",
        ),
    ],
)
def test_bad_plans_are_config_errors(
    make_settings: SettingsFactory, real_providers: Providers, override: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        resolve_plan(make_settings(override), real_providers, dataset="x")


def test_requires_out_of_stage_order_is_a_config_error(
    make_settings: SettingsFactory, real_providers: Providers
) -> None:
    providers = {**real_providers, "lift/backwards": BackwardsLift}
    with pytest.raises(ConfigError, match="lift/backwards"):
        resolve_plan(make_settings("extraction.plan.lift=backwards"), providers, dataset="x")


def test_a_missing_settings_section_is_a_config_error(
    make_settings: SettingsFactory, real_providers: Providers
) -> None:
    settings = make_settings()
    sections = {k: v for k, v in settings.sections.items() if k != "size_priors"}
    with pytest.raises(ConfigError, match=r"\[size_priors\]"):
        resolve_plan(replace(settings, sections=sections), real_providers, dataset="x")


def test_provider_overrides_and_settings_sections_reach_params(
    make_settings: SettingsFactory, real_providers: Providers
) -> None:
    settings = make_settings(
        "extraction.providers.kinematics.savgol.window=9", "size_priors.ball.mean_m=0.2"
    )
    plan = resolve_plan(settings, real_providers, dataset="x")
    assert plan.step(Stage.KINEMATICS).params.window == 9  # type: ignore[attr-defined]
    priors = plan.step(Stage.SCALE).params.priors  # type: ignore[attr-defined]
    assert isinstance(priors, SizePriorTable)
    assert priors.root["ball"].mean_m == 0.2


def test_sampling_from_flags_and_per_dataset_fps(
    make_settings: SettingsFactory, real_providers: Providers
) -> None:
    settings = make_settings()
    assert (
        resolve_plan(settings, real_providers, dataset="kubric_synthetic").sampling.target_fps == 24
    )
    assert (
        resolve_plan(settings, real_providers, dataset="synthetic_fixture").sampling.target_fps
        is None
    )
    flagged = resolve_plan(
        settings, real_providers, dataset="kubric_synthetic", fps=5.0, max_side=320
    ).sampling
    assert (flagged.target_fps, flagged.max_side) == (5.0, 320)
    stopped = resolve_plan(
        make_settings("extraction.sampling.stop=6"), real_providers, dataset="x"
    ).sampling
    assert stopped.stop == 6
