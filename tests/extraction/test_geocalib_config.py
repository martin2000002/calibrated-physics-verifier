"""An estimated GeoCalib focal is only configured with its measured floor."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest
from support import SettingsFactory
from wiring import resolve_plan

from physverify.extraction._base import StageProvider
from physverify.extraction.plan import ResolvedPlan
from physverify.schema.stages import Stage


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
def test_the_precise_profile_is_the_gate_commit(
    make_settings: SettingsFactory,
    real_providers: Mapping[str, type[StageProvider]],
    dataset: str,
) -> None:
    """Camera geocalib with the assumed focal and 8 frames; the LIFT temporal filter off."""
    plan = resolve_plan(make_settings(profile="precise"), real_providers, dataset=dataset)
    camera = plan.step(Stage.CAMERA)
    lift = plan.step(Stage.LIFT).params
    assert camera.provider.name == "geocalib"
    assert (camera.params.focal, camera.params.n_frames) == ("assumed", 8)  # type: ignore[attr-defined]
    assert lift.temporal_filter == "none"  # type: ignore[attr-defined]


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
