import math
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest
from settings_factory import SettingsFactory

from physverify.app import plugins
from physverify.checks.r5a_fall_shape import FallShapeCheck
from physverify.checks.variants._whitened import WhitenedFallShapeParams
from physverify.checks.variants.r5a_line_of_sight import LineOfSightFallShapeCheck
from physverify.physics.geometry import tilted
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.constants import STANDARD_GRAVITY

G = STANDARD_GRAVITY
CHI2_4_Q99 = 13.276704135987622


def only(case: sc.Case) -> CheckResult:
    results = LineOfSightFallShapeCheck(WhitenedFallShapeParams()).evaluate(sc.bare_inputs(case))
    assert len(results) == 1
    return results[0]


def with_gravity(case: sc.Case, gravity: np.ndarray, sigma: float = 0.0) -> sc.Case:
    states = replace(case.states, gravity_dir=gravity, gravity_dir_sigma=sigma)
    return replace(case, states=states)


def turned(case: sc.Case, radians: float) -> sc.Case:
    return with_gravity(case, tilted(case.states.gravity_dir, np.array([[radians, 0.0]]))[0])


def test_it_is_the_selectable_los_tilt_variant_of_r5a(make_settings: SettingsFactory) -> None:
    variant = plugins.build("check_variants")["R5a/los_tilt"]
    assert variant is LineOfSightFallShapeCheck
    assert variant.requires == FallShapeCheck.requires
    assert variant.magnitude_unit == "accel_tilt_chi2_4"
    assert variant.perturbation_unit == FallShapeCheck.magnitude_unit
    chosen = plugins.configured("checks", make_settings("check_variants.R5a=los_tilt"))
    assert chosen["R5a"] is LineOfSightFallShapeCheck
    assert plugins.configured("checks", make_settings(profile="precise"))["R5a"] is not variant


def test_a_fall_along_gravity_does_not_tilt_and_a_turned_gravity_does() -> None:
    fall = sc.with_sigma(sc.parabola(pitch_deg=25.0), position=0.002)
    clean, off = only(fall), only(turned(fall, math.radians(10.0)))
    assert clean.magnitude == pytest.approx(0.0, abs=1e-6)
    assert off.magnitude > 100 * CHI2_4_Q99
    assert sum(off.details["tilt_per_window"]) == pytest.approx(off.magnitude)
    assert off.details["accel_rel_diff_estimated"] == pytest.approx(0.0, abs=1e-6)


def test_a_gravity_sigma_absorbs_a_tilt_of_its_size() -> None:
    fall = turned(sc.with_sigma(sc.parabola(), position=0.002), math.radians(2.0))
    sure = only(fall).magnitude
    unsure = only(with_gravity(fall, fall.states.gravity_dir, math.radians(5.0))).magnitude
    assert unsure < CHI2_4_Q99 < sure


def test_exact_positions_are_weighted_with_the_floor_and_still_score() -> None:
    fall = only(sc.parabola())
    off = only(turned(sc.parabola(), math.radians(1.0)))
    assert math.isfinite(fall.magnitude) and fall.sigma == 0.0
    assert off.magnitude > fall.magnitude + 100 * CHI2_4_Q99


@pytest.mark.parametrize("sigma", [0.0, math.nan])
def test_an_unknown_gravity_or_gravity_sigma_abstains_as_unknown_uncertainty(sigma: float) -> None:
    case = sc.with_sigma(sc.parabola(), position=0.002)
    gravity = np.full(3, np.nan) if sigma == 0.0 else case.states.gravity_dir
    result = only(with_gravity(case, gravity, sigma))
    assert result.abstain_reason is AbstainReason.UNKNOWN_UNCERTAINTY


def test_a_flight_row_without_sigma_abstains_as_unknown_uncertainty() -> None:
    case = sc.with_sigma(sc.parabola(), position=0.002)
    result = only(sc.with_values(case, "position_sigma", "ball", [10], np.nan))
    assert result.abstain_reason is AbstainReason.UNKNOWN_UNCERTAINTY
