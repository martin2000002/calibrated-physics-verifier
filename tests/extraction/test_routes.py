"""Route plans substitute providers in the main plan: the stages upstream of a route's first
change keep the main plan's cache keys, and the route's own profile entries move no main key."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import pytest
from settings_factory import SettingsFactory
from support import chain_keys, entry, profile_of
from synthetic_clip import SyntheticClip
from wiring import resolve_plan

from physverify.extraction._base import StageProvider
from physverify.extraction.plan import ResolvedPlan
from physverify.extraction.routes import extra_plans, route_plans, with_providers
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.stages import STAGE_ORDER, Stage

Providers = Mapping[str, type[StageProvider]]
AREA = {Stage.LIFT: "backproject_area_depth"}
HEAVY = [Stage.DETECT, Stage.TRACK, Stage.DEPTH, Stage.CAMERA]
FROM_LIFT = [Stage.LIFT, Stage.SCALE, Stage.KINEMATICS, Stage.CONTACTS]
DATASET = "synthetic_fixture"
MAIN_ENTRIES = (
    entry(Stage.TRACK, "edgetam", "centroid_px_rel", 0.05),
    entry(Stage.DEPTH, "vda_small", "depth_rel", 0.1),
    entry(Stage.LIFT, "backproject", "centroid_rel", 0.1),
    entry(Stage.LIFT, "backproject", "size_rel", 0.08),
)
AREA_ENTRIES = (
    entry(Stage.LIFT, "backproject_area_depth", "centroid_rel", 0.3),
    entry(Stage.LIFT, "backproject_area_depth", "size_rel", 0.2),
)


def keys_of(plan: ResolvedPlan, clip: SyntheticClip, profile: ErrorProfile) -> dict[Stage, str]:
    return chain_keys(plan, clip.manifest, gt_version="v1", profile=profile)


@pytest.mark.parametrize("profile", ["fast", "precise"])
def test_a_lift_route_shares_the_heavy_keys_and_changes_the_light_ones(
    make_settings: SettingsFactory,
    real_providers: Providers,
    fixture_clip: SyntheticClip,
    profile: str,
) -> None:
    settings = make_settings(profile=profile)
    resolve = partial(resolve_plan, providers=real_providers, dataset=DATASET)
    main = resolve(settings)
    route = route_plans(settings, {"area": AREA}, resolve)["area"]
    assert dict(route.names) == {**main.names, **AREA}
    for before, after in zip(main.stages, route.stages, strict=True):
        if before.stage is not Stage.LIFT:
            assert after == before
    errors = profile_of(*MAIN_ENTRIES, *AREA_ENTRIES)
    main_keys, route_keys = (
        keys_of(main, fixture_clip, errors),
        keys_of(route, fixture_clip, errors),
    )
    assert [s for s in STAGE_ORDER if main_keys[s] == route_keys[s]] == HEAVY
    assert [s for s in STAGE_ORDER if main_keys[s] != route_keys[s]] == FROM_LIFT


@pytest.mark.parametrize("profile", ["fast", "precise"])
def test_the_routes_own_profile_entries_leave_every_main_key_as_it_was(
    make_settings: SettingsFactory,
    real_providers: Providers,
    fixture_clip: SyntheticClip,
    profile: str,
) -> None:
    main = resolve_plan(make_settings(profile=profile), real_providers, dataset=DATASET)
    before = keys_of(main, fixture_clip, profile_of(*MAIN_ENTRIES))
    after = keys_of(main, fixture_clip, profile_of(*MAIN_ENTRIES, *AREA_ENTRIES))
    assert after == before


def test_a_route_over_a_true_stage_is_the_main_plan(
    make_settings: SettingsFactory, real_providers: Providers
) -> None:
    settings = make_settings()
    resolve = partial(
        resolve_plan, providers=real_providers, dataset=DATASET, gt_stages=(Stage.LIFT,)
    )
    main = resolve(settings)
    routes = route_plans(settings, {"area": AREA, "same": {}}, resolve)
    assert routes == {"area": main, "same": main}
    assert extra_plans(main, routes) == ()


def test_extra_plans_are_distinct_and_in_route_order(
    make_settings: SettingsFactory, real_providers: Providers
) -> None:
    settings = make_settings()
    resolve = partial(resolve_plan, providers=real_providers, dataset=DATASET)
    routes = route_plans(settings, {"b": AREA, "a": AREA, "main": {}}, resolve)
    assert extra_plans(resolve(settings), routes) == (routes["a"],)


def test_with_providers_changes_only_the_listed_stages(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    changed = with_providers(settings, AREA)
    plan = dict(settings.sections["extraction"]["plan"])
    assert changed.sections["extraction"]["plan"] == {**plan, "lift": "backproject_area_depth"}
    assert {k: v for k, v in changed.sections.items() if k != "extraction"} == {
        k: v for k, v in settings.sections.items() if k != "extraction"
    }
