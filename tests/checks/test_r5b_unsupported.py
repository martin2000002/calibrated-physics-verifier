import dataclasses
import math

import check_scenarios as sc
import pytest
from settings_factory import SettingsFactory

from physverify.app import plugins
from physverify.checks._base import CheckInputs
from physverify.checks.r5b_gravity_value import GravityValueCheck, GravityValueParams
from physverify.checks.variants.r5b_unsupported import UnsupportedGravityValueCheck
from physverify.schema.checks import AbstainReason, CheckResult, Verdict

Z_975 = 1.959963984540054
GAP_BOUND = Z_975 * 0.05


def inputs(case: sc.Case, bound: float) -> CheckInputs:
    return dataclasses.replace(sc.bare_inputs(case), support_gap_bound=bound)


def variant(case: sc.Case, bound: float = GAP_BOUND) -> list[CheckResult]:
    return UnsupportedGravityValueCheck(GravityValueParams()).evaluate(inputs(case, bound))


def default(case: sc.Case) -> list[CheckResult]:
    return GravityValueCheck(GravityValueParams()).evaluate(sc.bare_inputs(case))


def dicts(results: list[CheckResult]) -> list[dict[str, object]]:
    return [r.to_dict() for r in results]


def test_it_is_the_selectable_unsupported_variant_of_r5b(make_settings: SettingsFactory) -> None:
    chosen = plugins.build("check_variants")["R5b/unsupported"]
    assert chosen is UnsupportedGravityValueCheck
    assert chosen.requires == GravityValueCheck.requires
    assert chosen.magnitude_unit == GravityValueCheck.magnitude_unit
    selected = plugins.configured("checks", make_settings("check_variants.R5b=unsupported"))
    assert selected["R5b"] is UnsupportedGravityValueCheck
    assert plugins.configured("checks", make_settings())["R5b"] is GravityValueCheck


def test_an_unmarked_roll_on_the_scene_plane_is_not_a_flight() -> None:
    case = sc.unmarked_roll()
    assert default(case)[0].magnitude == pytest.approx(1.0)
    assert variant(case) == []


def test_a_genuine_flight_beside_a_resting_box_is_scored_as_by_default() -> None:
    case = sc.flight_beside_box()
    assert dicts(variant(case)) == dicts(default(case))
    assert dicts(variant(case, bound=0.0)) == dicts(default(case))


def test_a_body_gliding_above_the_plane_without_acceleration_is_still_scored() -> None:
    case = sc.gliding_ball()
    (result,) = variant(case)
    assert result.verdict is Verdict.UNSCORED
    assert result.magnitude == pytest.approx(1.0)
    assert dicts([result]) == dicts(default(case))


def test_flights_without_a_scene_plane_are_kept() -> None:
    case = sc.parabola(pitch_deg=25.0)
    assert dicts(variant(case)) == dicts(default(case))


def test_an_unknown_gap_bound_abstains_when_there_are_flights() -> None:
    (result,) = variant(sc.unmarked_roll(), bound=math.nan)
    assert result.verdict is Verdict.ABSTAIN
    assert result.abstain_reason is AbstainReason.UNKNOWN_UNCERTAINTY
    assert result.details["flights"] == 1
    assert result.details["support_gap_bound"] is None
