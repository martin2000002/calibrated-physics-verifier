"""A route that gives R3b ``lift/backproject_contour_circle`` changes R3b's inputs alone: the main
plan, its cache keys and every other check's inputs stay as they are."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from functools import partial

import check_scenarios as sc
import pytest
from settings_factory import SettingsFactory
from support import chain_keys, entry, profile_of
from test_check_routes import AREA_LIFT, DATASET, PlanSource, lift_by_check, source_for
from wiring import resolve_plan

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.extraction.plan import ResolvedPlan
from physverify.schema.clip import ClipManifest
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.verification.check_routes import (
    RoutedSource,
    check_routing,
    input_plans_by_route,
    routed_inputs,
)
from physverify.verification.inputs import ClipInputs, InputSource

CC_LIFT = "backproject_contour_circle"
CC_ROUTE = (
    f"verification.routes.contour_centre.lift={CC_LIFT}",
    "verification.check_routes.R3b=contour_centre",
)
PROFILES = ("fast", "precise")
HEAVY = [Stage.DETECT, Stage.TRACK, Stage.DEPTH, Stage.CAMERA]
FROM_LIFT = [Stage.LIFT, Stage.SCALE, Stage.KINEMATICS, Stage.CONTACTS]
MAIN_ENTRIES = (
    entry(Stage.TRACK, "edgetam", "centroid_px_rel", 0.05),
    entry(Stage.DEPTH, "vda_small", "depth_rel", 0.1),
    entry(Stage.LIFT, "backproject", "centroid_rel", 0.1),
    entry(Stage.LIFT, "backproject", "size_rel", 0.08),
)
CC_ENTRIES = tuple(
    entry(Stage.LIFT, CC_LIFT, quantity, 0.3)
    for quantity in ("centroid_rel", "size_rel", "orientation", "support_gap_rel")
)


def resolver(gt_stages: Sequence[Stage] = ()) -> partial[ResolvedPlan]:
    return partial(
        resolve_plan,
        providers=PLUGIN_KINDS["providers"](),
        dataset=DATASET,
        gt_stages=tuple(gt_stages),
    )


def plans(settings: Settings) -> tuple[ResolvedPlan, dict[str, ResolvedPlan]]:
    return input_plans_by_route(settings, resolver(), PLUGIN_KINDS["checks"]())


def keys_of(plan: ResolvedPlan, profile: ErrorProfile) -> dict[Stage, str]:
    return chain_keys(plan, sc.parabola(clip_id="c").clip, gt_version="v1", profile=profile)


@pytest.mark.parametrize("profile", PROFILES)
def test_the_command_line_route_adds_to_the_profiles_routes(
    make_settings: SettingsFactory, profile: str
) -> None:
    profiled = check_routing(make_settings(profile=profile)).check_routes
    routed = check_routing(make_settings(*CC_ROUTE, profile=profile)).check_routes
    assert routed == {**profiled, "R3b": "contour_centre"}
    assert profiled == ({"R3a": "area_depth"} if profile == "precise" else {})


@pytest.mark.parametrize("profile", PROFILES)
def test_the_route_leaves_the_main_plan_and_its_keys_and_shares_the_heavy_keys(
    make_settings: SettingsFactory, profile: str
) -> None:
    errors = profile_of(*MAIN_ENTRIES, *CC_ENTRIES)
    main, before = plans(make_settings(profile=profile))
    routed_main, routes = plans(make_settings(*CC_ROUTE, profile=profile))
    assert routed_main == main
    assert keys_of(routed_main, errors) == keys_of(main, errors)
    assert {name: plan for name, plan in routes.items() if name != "contour_centre"} == before
    route = routes["contour_centre"]
    assert dict(route.names) == {**main.names, Stage.LIFT: CC_LIFT}
    main_keys, route_keys = (
        keys_of(main, errors),
        keys_of(route, errors),
    )
    assert [s for s in STAGE_ORDER if main_keys[s] == route_keys[s]] == HEAVY
    assert [s for s in STAGE_ORDER if main_keys[s] != route_keys[s]] == FROM_LIFT


class RecordingSource:
    """``PlanSource`` that keeps every item it yields."""

    def __init__(self, plan: ResolvedPlan) -> None:
        self.source = PlanSource.of(plan)
        self.items: list[ClipInputs] = []

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        for item in self.source.load(clips):
            self.items.append(item)
            yield item


def test_r3b_alone_reads_the_route_and_every_other_check_the_main_item(
    make_settings: SettingsFactory,
) -> None:
    built: dict[str, RecordingSource] = {}

    def source_of(plan: ResolvedPlan) -> InputSource:
        built[plan.names[Stage.LIFT]] = RecordingSource(plan)
        return built[plan.names[Stage.LIFT]]

    settings = make_settings(*CC_ROUTE, profile="precise")
    source = routed_inputs(settings, resolver(), PLUGIN_KINDS["checks"](), source_of)
    assert isinstance(source, RoutedSource)
    (item,) = source.load([sc.parabola(clip_id="c").clip])
    (main,) = built["backproject"].items
    (route,) = built[CC_LIFT].items
    (area,) = built[AREA_LIFT].items
    check_ids = [check.check_id for check in PLUGIN_KINDS["checks"]().values()]
    assert len(check_ids) == 9
    for check_id in check_ids:
        expected = {"R3a": area, "R3b": route}.get(check_id, main)
        assert item.for_check(check_id).inputs is expected.inputs, check_id


@pytest.mark.parametrize("profile", PROFILES)
def test_the_route_lifts_r3b_alone(make_settings: SettingsFactory, profile: str) -> None:
    lifts = lift_by_check(make_settings(*CC_ROUTE, profile=profile))
    assert lifts.pop("R3b") == CC_LIFT
    if profile == "precise":
        assert lifts.pop("R3a") == AREA_LIFT
    assert set(lifts.values()) == {"backproject"}


@pytest.mark.parametrize("profile", PROFILES)
def test_under_truth_at_every_stage_the_main_source_is_loaded_alone(
    make_settings: SettingsFactory, profile: str
) -> None:
    settings = make_settings(*CC_ROUTE, profile=profile)
    resolve = resolver(gt_stages=STAGE_ORDER)
    source = routed_inputs(settings, resolve, PLUGIN_KINDS["checks"](), PlanSource.of)
    assert isinstance(source, PlanSource)
    assert source.names == dict(resolve(settings).names)


@pytest.mark.parametrize("profile", PROFILES)
def test_the_routes_profile_entries_leave_every_main_key_as_it_was(
    make_settings: SettingsFactory, profile: str
) -> None:
    main, routes = plans(make_settings(*CC_ROUTE, profile=profile))
    before = keys_of(main, profile_of(*MAIN_ENTRIES))
    after = keys_of(main, profile_of(*MAIN_ENTRIES, *CC_ENTRIES))
    assert after == before
    route = routes["contour_centre"]
    route_before = keys_of(route, profile_of(*MAIN_ENTRIES))
    route_after = keys_of(route, profile_of(*MAIN_ENTRIES, *CC_ENTRIES))
    assert route_after[Stage.LIFT] != route_before[Stage.LIFT]


@pytest.mark.parametrize("profile", PROFILES)
def test_the_size_guard_accepts_the_route_and_still_refuses_r4_size_on_area_depth(
    make_settings: SettingsFactory, profile: str
) -> None:
    assert isinstance(source_for(make_settings(*CC_ROUTE, profile=profile)), RoutedSource)
    on_cc = make_settings(*CC_ROUTE, "verification.check_routes.R4_size=contour_centre")
    assert isinstance(source_for(on_cc), RoutedSource)
    on_area = (
        *CC_ROUTE,
        f"verification.routes.area_depth.lift={AREA_LIFT}",
        "verification.check_routes.R4_size=area_depth",
    )
    with pytest.raises(ConfigError, match=f"R4_size would read lift/{AREA_LIFT}"):
        source_for(make_settings(*on_area, profile=profile))


def test_no_profile_routes_a_check_to_the_contour_circle_lift(
    make_settings: SettingsFactory,
) -> None:
    for profile in PROFILES:
        routing = check_routing(make_settings(profile=profile))
        routed: Mapping[str, Mapping[Stage, str]] = {
            check: routing.routes[route] for check, route in routing.check_routes.items()
        }
        assert all(CC_LIFT not in stages.values() for stages in routed.values())
