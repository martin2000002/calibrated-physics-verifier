"""An estimated GeoCalib focal is only configured with its measured floor; the committed precise
camera, lift and size spread."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest
from settings_factory import SettingsFactory
from wiring import resolve_plan

from physverify.config.models import SIZE_PRIORS_SECTION, SizePriorTable
from physverify.extraction._base import StageProvider
from physverify.extraction.plan import ResolvedPlan
from physverify.schema.stages import Stage
from physverify.sizing.priors import GENERIC_PRIOR


def floor_missing(plan: ResolvedPlan) -> bool:
    """True when CAMERA is geocalib with focal = "geocalib" and no finite floor > 0."""
    step = plan.step(Stage.CAMERA)
    if step.provider.name != "geocalib":
        return False
    params = step.params
    floor = params.focal_sigma_floor_rel  # type: ignore[attr-defined]
    return params.focal == "geocalib" and not (math.isfinite(floor) and floor > 0)  # type: ignore[attr-defined]


@pytest.mark.parametrize("profile", ["fast", "precise"])
@pytest.mark.parametrize("dataset", ["kubric_synthetic", "physics_iq", "pisa_real"])
def test_b4_the_committed_configuration_never_uses_the_model_sigma_alone(
    make_settings: SettingsFactory,
    real_providers: Mapping[str, type[StageProvider]],
    profile: str,
    dataset: str,
) -> None:
    settings = make_settings(profile=profile)
    assert not floor_missing(resolve_plan(settings, real_providers, dataset=dataset))


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        (("extraction.plan.camera=geocalib",), True),
        (
            (
                "extraction.plan.camera=geocalib",
                "extraction.providers.camera.geocalib.focal=assumed",
            ),
            False,
        ),
        (
            (
                "extraction.plan.camera=geocalib",
                "extraction.providers.camera.geocalib.focal_sigma_floor_rel=0.1",
            ),
            False,
        ),
        (
            (
                "extraction.plan.camera=geocalib",
                "extraction.providers.camera.geocalib.focal_sigma_floor_rel=0.0",
            ),
            True,
        ),
    ],
)
def test_b4_the_rule_flags_an_estimated_focal_without_a_floor(
    make_settings: SettingsFactory,
    real_providers: Mapping[str, type[StageProvider]],
    overrides: tuple[str, ...],
    missing: bool,
) -> None:
    # fast: precise pins its own geocalib table, which overrides these keys
    settings = make_settings(*overrides, profile="fast")
    assert (
        floor_missing(resolve_plan(settings, real_providers, dataset="kubric_synthetic")) is missing
    )


@pytest.mark.parametrize("dataset", ["kubric_synthetic", "physics_iq", "pisa_real"])
def test_the_precise_profile_takes_the_clip_focal_and_keeps_the_v5_lift(
    make_settings: SettingsFactory,
    real_providers: Mapping[str, type[StageProvider]],
    dataset: str,
) -> None:
    """Camera moge2_focal on 8 frames with GeoCalib gravity (ADR 0027); LIFT backproject v5 with
    its temporal filter off (lift/ground_anchored is opt-in, ADR 0028)."""
    plan = resolve_plan(make_settings(profile="precise"), real_providers, dataset=dataset)
    camera = plan.step(Stage.CAMERA)
    lift = plan.step(Stage.LIFT)
    assert camera.provider.name == "moge2_focal"
    assert (camera.params.n_frames, camera.params.gravity) == (8, "geocalib")  # type: ignore[attr-defined]
    assert lift.provider.name == "backproject"
    assert lift.params.temporal_filter == "none"  # type: ignore[attr-defined]


@pytest.mark.parametrize(("profile", "spread"), [("fast", 0.85), ("precise", 1.435)])
def test_the_generic_size_spread_follows_the_profile_camera(
    make_settings: SettingsFactory, profile: str, spread: float
) -> None:
    """1.435 goes only with camera/moge2_focal (ADR 0027); the assumed camera keeps 0.85."""
    priors = make_settings(profile=profile).section(SIZE_PRIORS_SECTION, SizePriorTable).root
    assert (priors[GENERIC_PRIOR].mean_m, priors[GENERIC_PRIOR].rel_spread) == (0.19, spread)


@pytest.mark.parametrize("dataset", ["kubric_synthetic", "physics_iq", "pisa_real"])
def test_the_precise_profile_is_the_base_chain(
    make_settings: SettingsFactory,
    real_providers: Mapping[str, type[StageProvider]],
    dataset: str,
) -> None:
    """Contacts stay proximity and depth stays vda_small in the precise base chain."""
    plan = resolve_plan(make_settings(profile="precise"), real_providers, dataset=dataset)
    assert plan.step(Stage.DEPTH).provider.name == "vda_small"
    assert plan.step(Stage.CONTACTS).provider.name == "proximity"
