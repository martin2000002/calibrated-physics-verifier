import math
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest
from settings_factory import SettingsFactory

from physverify.app import plugins
from physverify.checks.r5b_gravity_value import GravityValueCheck, GravityValueParams
from physverify.checks.variants.r5b_line_of_sight import (
    LineOfSightGravityValueCheck,
    LineOfSightGravityValueParams,
)
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.constants import STANDARD_GRAVITY

G = STANDARD_GRAVITY


def only(case: sc.Case) -> CheckResult:
    check = LineOfSightGravityValueCheck(LineOfSightGravityValueParams())
    (result,) = check.evaluate(sc.bare_inputs(case))
    return result


def default(case: sc.Case) -> CheckResult:
    (result,) = GravityValueCheck(GravityValueParams()).evaluate(sc.bare_inputs(case))
    return result


def test_it_is_the_selectable_los_ballistic_variant_of_r5b(make_settings: SettingsFactory) -> None:
    chosen = plugins.build("check_variants")["R5b/los_ballistic"]
    assert chosen is LineOfSightGravityValueCheck
    assert chosen.requires == GravityValueCheck.requires
    assert chosen.magnitude_unit == GravityValueCheck.magnitude_unit
    selected = plugins.configured("checks", make_settings("check_variants.R5b=los_ballistic"))
    assert selected["R5b"] is LineOfSightGravityValueCheck
    assert plugins.configured("checks", make_settings(profile="precise"))["R5b"] is not chosen


@pytest.mark.parametrize("g", [G, 1.2 * G, 0.7 * G])
def test_it_reads_g_along_gravity_as_the_default_does(g: float) -> None:
    case = sc.with_sigma(sc.parabola(g=g, pitch_deg=25.0), position=0.002)
    result = only(case)
    assert result.magnitude == pytest.approx(abs(g / G - 1.0), abs=1e-9)
    assert result.magnitude == pytest.approx(default(case).magnitude, abs=1e-9)
    assert result.sigma > 0.0
    assert result.details["used_flights"] == 1


def test_exact_positions_are_weighted_with_the_floor() -> None:
    result = only(sc.parabola(g=1.2 * G))
    assert result.magnitude == pytest.approx(0.2, abs=1e-9)
    assert math.isfinite(result.sigma)


def test_a_scale_sigma_adds_its_shared_term() -> None:
    case = sc.with_sigma(sc.parabola(g=1.2 * G), position=0.002)
    loose = replace(case, scale=sc.oracle_scale(rel_sigma=0.05))
    assert only(loose).sigma == pytest.approx(math.hypot(only(case).sigma, 1.2 * 0.05))


def test_a_flight_row_without_sigma_is_fitted_with_an_unknown_sigma() -> None:
    case = sc.with_sigma(sc.parabola(g=1.2 * G), position=0.002)
    result = only(sc.with_values(case, "position_sigma", "ball", [10], np.nan))
    assert result.magnitude == pytest.approx(0.2, abs=1e-9)
    assert math.isnan(result.sigma)


def test_an_unknown_gravity_abstains_as_unknown_uncertainty() -> None:
    case = sc.with_sigma(sc.parabola(), position=0.002)
    states = replace(case.states, gravity_dir=np.full(3, np.nan))
    result = only(replace(case, states=states))
    assert result.abstain_reason is AbstainReason.UNKNOWN_UNCERTAINTY
