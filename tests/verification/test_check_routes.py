"""Input routes per check: a routed check reads its route's inputs, every other check the main
plan's, and a size-change check never reads a depth taken from the apparent size."""

from __future__ import annotations

import importlib
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from pathlib import Path

import check_scenarios as sc
import pytest
from settings_factory import SettingsFactory
from wiring import resolve_plan

from physverify.app import plugins
from physverify.app.plugins import PLUGIN_KINDS
from physverify.checks._base import Check, CheckInputs, unscored
from physverify.checks._params import CommonParams
from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.core.registry import discover
from physverify.extraction._base import StageProvider, provider_key, validate_provider
from physverify.extraction.plan import ResolvedPlan
from physverify.extraction.routes import route_plans
from physverify.schema.checks import CheckResult
from physverify.schema.clip import ClipManifest
from physverify.schema.stages import Stage
from physverify.verification.check_routes import (
    VERIFICATION_SECTION,
    RoutedSource,
    check_routing,
    routed_source,
)
from physverify.verification.check_set import (
    CHECKS_SECTION,
    ChecksSettings,
    active_checks_hash,
    checks_hash,
)
from physverify.verification.inputs import ClipInputs, InputSource, provenance
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import VerificationRunner, VerificationSettings
from physverify.verification.thresholding import ThresholdTable

AREA_LIFT = "backproject_area_depth"
ROUTE_AREA = (f"verification.routes.area_depth.lift={AREA_LIFT}",)
ROUTED_R3A = (*ROUTE_AREA, "verification.check_routes.R3a=area_depth")
DATASET = "synthetic_fixture"
Providers = Mapping[str, type[StageProvider]]


class PlanSource:
    """Inputs whose provenance names the stage providers ``names``."""

    def __init__(self, names: Mapping[Stage, str]) -> None:
        self.names = dict(names)

    @classmethod
    def of(cls, plan: ResolvedPlan) -> PlanSource:
        return cls(plan.names)

    def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
        for clip in clips:
            case = sc.parabola(clip_id=clip.clip_id)
            inputs = case.inputs(provenance=provenance(self.names, None))
            yield ClipInputs(case.clip, inputs, None)


def source_for(settings: Settings, providers: Providers | None = None) -> InputSource:
    registry = PLUGIN_KINDS["providers"]() if providers is None else providers
    resolve = partial(resolve_plan, providers=registry, dataset=DATASET)
    routing = check_routing(settings)
    return routed_source(
        resolve(settings),
        route_plans(settings, routing.routes, resolve),
        routing.check_routes,
        PlanSource.of,
    )


def lift_by_check(settings: Settings, providers: Providers | None = None) -> dict[str, str]:
    (item,) = source_for(settings, providers).load([sc.parabola(clip_id="c").clip])
    found: dict[str, str] = {}
    for check in plugins.configured("checks", settings).values():
        inputs = item.for_check(check.check_id).inputs
        assert inputs is not None
        found[check.check_id] = inputs.provenance[Stage.LIFT.value]
    return found


@pytest.mark.parametrize("overrides", [(), ROUTED_R3A])
def test_under_precise_only_r3a_reads_the_area_depth_lift(
    make_settings: SettingsFactory, overrides: tuple[str, ...]
) -> None:
    lifts = lift_by_check(make_settings(*overrides, profile="precise"))
    assert len(lifts) == 9
    assert lifts.pop("R3a") == AREA_LIFT
    assert set(lifts.values()) == {"backproject"}


def test_without_routes_every_check_reads_the_main_lift(make_settings: SettingsFactory) -> None:
    settings = make_settings(profile="fast")
    assert check_routing(settings).check_routes == {}
    assert isinstance(source_for(settings), PlanSource)
    assert set(lift_by_check(settings).values()) == {"backproject"}


def test_a_route_that_resolves_to_the_main_plan_is_not_loaded_again(
    make_settings: SettingsFactory,
) -> None:
    settings = make_settings(*ROUTED_R3A)
    resolve = partial(
        resolve_plan,
        providers=PLUGIN_KINDS["providers"](),
        dataset=DATASET,
        gt_stages=(Stage.LIFT,),
    )
    routing = check_routing(settings)
    routes = route_plans(settings, routing.routes, resolve)
    source = routed_source(resolve(settings), routes, routing.check_routes, PlanSource.of)
    assert isinstance(source, PlanSource)


def spy(seen: dict[str, str]) -> Callable[[str], type[Check]]:
    def make(check_id: str) -> type[Check]:
        def evaluate(self: Check, inputs: CheckInputs) -> Sequence[CheckResult]:
            seen[check_id] = inputs.provenance[Stage.LIFT.value]
            return [unscored(type(self), inputs, ("a",), 0.1, 0.01, (0, 3), {})]

        def init(self: Check, params: CommonParams | None = None) -> None:
            return None

        namespace = {
            "check_id": check_id,
            "requires": frozenset(),
            "magnitude_unit": "unit",
            "Params": CommonParams,
            "__init__": init,
            "evaluate": evaluate,
        }
        return type(f"Spy{check_id}", (), namespace)

    return make


def test_the_runner_hands_each_check_its_routed_inputs(make_settings: SettingsFactory) -> None:
    seen: dict[str, str] = {}
    checks = [spy(seen)(check_id)() for check_id in ("R3a", "R1", "R4_size")]
    runner = VerificationRunner(checks, RawPolicy(), ThresholdTable.provisional({}))
    source = source_for(make_settings(*ROUTED_R3A, profile="precise"))
    report = runner.run(source, [sc.parabola(clip_id="c").clip])
    assert seen == {"R3a": AREA_LIFT, "R1": "backproject", "R4_size": "backproject"}
    assert {r.check_id: r.provenance[Stage.LIFT.value] for r in report.results} == seen


def test_routed_inputs_must_match_the_main_clip() -> None:
    plan_source = PlanSource({Stage.LIFT: "backproject"})

    class Shifted:
        def load(self, clips: Sequence[ClipManifest]) -> Iterator[ClipInputs]:
            return plan_source.load([sc.parabola(clip_id="other").clip for _ in clips])

    routed = RoutedSource(plan_source, {"r": Shifted()}, {"R3a": "r"})
    with pytest.raises(ValueError, match="other"):
        list(routed.load([sc.parabola(clip_id="c").clip]))


FORBIDDEN = (
    (*ROUTE_AREA, "verification.check_routes.R4_size=area_depth"),
    (
        f"verification.routes.renamed.lift={AREA_LIFT}",
        "verification.check_routes.R4_size=renamed",
    ),
)


@pytest.mark.parametrize("overrides", FORBIDDEN)
def test_r4_size_on_the_area_depth_lift_is_refused_from_the_command_line(
    make_settings: SettingsFactory, overrides: tuple[str, ...]
) -> None:
    for profile in ("fast", "precise"):
        settings = make_settings(*overrides, profile=profile)
        with pytest.raises(ConfigError, match="R4_size"):
            check_routing(settings)
        with pytest.raises(ConfigError, match="R4_size"):
            settings.section(VERIFICATION_SECTION, VerificationSettings)
        with pytest.raises(ConfigError, match="R4_size"):
            active_checks_hash(settings, plugins.configured("checks", settings))


FORBIDDEN_TOML = (
    f'\n[verification.routes.area_depth]\nlift = "{AREA_LIFT}"\n'
    '\n[verification.check_routes]\nR4_size = "area_depth"\n'
)


@pytest.mark.parametrize("file", ["verifier/checks.toml", "local.toml"])
def test_r4_size_on_the_area_depth_lift_is_refused_from_a_config_file(
    make_settings: SettingsFactory, config_dir: Path, file: str
) -> None:
    path = config_dir / file
    path.write_text(path.read_text(encoding="utf-8") + FORBIDDEN_TOML, encoding="utf-8")
    with pytest.raises(ConfigError, match="R4_size"):
        check_routing(make_settings(profile="precise"))


def test_an_unknown_route_is_refused(make_settings: SettingsFactory) -> None:
    with pytest.raises(ConfigError, match="unknown route"):
        check_routing(make_settings("verification.check_routes.R3a=missing"))


def test_no_r4_size_implementation_is_routed_under_either_profile(
    make_settings: SettingsFactory,
) -> None:
    every = {**PLUGIN_KINDS["checks"](), **PLUGIN_KINDS["check_variants"]()}
    sizes = [check for check in every.values() if check.check_id == "R4_size"]
    assert len(sizes) > 1
    for profile in ("fast", "precise"):
        settings = make_settings(*ROUTED_R3A, profile=profile)
        routing = check_routing(settings)
        assert routing.stages_of(["R4_size"]) == {}
        (item,) = source_for(settings).load([sc.parabola(clip_id="c").clip])
        inputs = item.for_check("R4_size").inputs
        assert inputs is not None
        assert inputs.provenance[Stage.LIFT.value] != AREA_LIFT


def test_routes_enter_the_checks_hash_only_when_a_check_is_routed(
    make_settings: SettingsFactory,
) -> None:
    registry: Mapping[str, type[Check]] = plugins.build("checks")
    plain = make_settings()
    declared = make_settings(*ROUTE_AREA)
    routed = make_settings(*ROUTED_R3A)
    assert active_checks_hash(declared, registry) == active_checks_hash(plain, registry)
    assert active_checks_hash(routed, registry) != active_checks_hash(plain, registry)
    without_r3a = {key: check for key, check in registry.items() if check.check_id != "R3a"}
    assert active_checks_hash(routed, without_r3a) == active_checks_hash(plain, without_r3a)
    overrides = routed.section(CHECKS_SECTION, ChecksSettings)
    assert checks_hash(registry, overrides) == active_checks_hash(plain, registry)
    assert checks_hash(registry, overrides, check_routing(routed)) == (
        active_checks_hash(routed, registry)
    )


TEST_LIFT = """
from typing import ClassVar

from physverify.extraction.providers._backproject import BackprojectLift


class RouteTestLift(BackprojectLift):
    name: ClassVar[str] = "route_test_lift"
    version: ClassVar[str] = "1"


PLUGINS = (RouteTestLift,)
"""
TEST_ROUTE = """
[verification.routes.route_test]
lift = "route_test_lift"

[verification.check_routes]
R3a = "route_test"
"""


def test_a_new_lift_is_routed_with_one_plugin_file_and_toml_alone(
    make_settings: SettingsFactory,
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = f"fx_route_{uuid.uuid4().hex[:8]}"
    (tmp_path / package).mkdir()
    (tmp_path / package / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / package / "lift_route_test.py").write_text(TEST_LIFT, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    found = discover(package, (StageProvider,), provider_key, validate_provider)
    local = config_dir / "local.toml"
    local.write_text(local.read_text(encoding="utf-8") + TEST_ROUTE, encoding="utf-8")
    settings = make_settings()
    lifts = lift_by_check(settings, {**PLUGIN_KINDS["providers"](), **found})
    assert lifts.pop("R3a") == "route_test_lift"
    assert set(lifts.values()) == {"backproject"}
