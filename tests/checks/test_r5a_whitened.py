import math

import check_scenarios as sc
import numpy as np
import pytest
from settings_factory import SettingsFactory

from physverify.app import plugins
from physverify.checks.r5a_fall_shape import FallShapeCheck
from physverify.checks.variants.r5a_whitened import (
    WhitenedFallShapeCheck,
    WhitenedFallShapeParams,
)
from physverify.schema.checks import AbstainReason, CheckResult
from physverify.schema.constants import STANDARD_GRAVITY

G = STANDARD_GRAVITY
CHI2_3_Q99 = 11.344866730144373


def run(
    case: sc.Case, params: WhitenedFallShapeParams | None = None
) -> dict[tuple[str, ...], CheckResult]:
    check = WhitenedFallShapeCheck(params or WhitenedFallShapeParams())
    return {r.subject: r for r in check.evaluate(sc.bare_inputs(case))}


def only(case: sc.Case, params: WhitenedFallShapeParams | None = None) -> CheckResult:
    results = run(case, params)
    assert len(results) == 1
    return next(iter(results.values()))


def test_it_is_the_selectable_whitened_variant_of_r5a(make_settings: SettingsFactory) -> None:
    variant = plugins.build("check_variants")["R5a/whitened"]
    assert variant is WhitenedFallShapeCheck
    assert variant.requires == FallShapeCheck.requires
    assert variant.magnitude_unit == "accel_wald_chi2_3"
    assert variant.perturbation_unit == FallShapeCheck.magnitude_unit
    chosen = plugins.configured("checks", make_settings("check_variants.R5a=whitened"))
    assert chosen["R5a"] is WhitenedFallShapeCheck
    precise = plugins.configured("checks", make_settings(profile="precise"))
    assert precise["R5a"] is WhitenedFallShapeCheck
    assert plugins.configured("checks", make_settings(profile="fast"))["R5a"] is FallShapeCheck
    restored = plugins.configured("checks", make_settings("check_variants.R5a=default"))
    assert restored["R5a"] is FallShapeCheck


def test_a_free_fall_is_consistent_and_a_gravity_switch_is_not() -> None:
    fall = only(sc.with_sigma(sc.parabola(pitch_deg=25.0), position=0.002))
    switch = only(sc.with_sigma(sc.parabola(g_after=1.3 * G, change_frame=18), position=0.002))
    assert fall.magnitude == pytest.approx(0.0, abs=1e-6)
    assert switch.magnitude > 100 * CHI2_3_Q99
    assert switch.details["mode"] == "split"
    assert switch.details["accel_rel_diff_estimated"] > 0.1
    assert sum(switch.details["wald_per_axis"]) == pytest.approx(switch.magnitude)


def test_a_uniform_sigma_scales_the_statistic_by_its_inverse_square() -> None:
    case = sc.parabola(g_after=1.3 * G, change_frame=18)
    narrow = only(sc.with_sigma(case, position=0.002)).magnitude
    wide = only(sc.with_sigma(case, position=0.004)).magnitude
    assert wide == pytest.approx(narrow / 4.0, rel=1e-6)


def test_exact_positions_are_weighted_with_the_floor_and_still_score() -> None:
    fall = only(sc.parabola())
    switch = only(sc.parabola(g_after=1.3 * G, change_frame=18))
    assert math.isfinite(fall.magnitude) and fall.sigma == 0.0
    assert switch.magnitude > fall.magnitude + 100 * CHI2_3_Q99


def test_rows_labelled_exact_in_a_perceived_window_dominate_it() -> None:
    noisy = sc.jittered(sc.with_sigma(sc.parabola(), position=0.002), 0.002, seed=0)
    labelled = sc.with_values(noisy, "position_sigma", "ball", range(6), 0.0)
    assert only(noisy).magnitude < CHI2_3_Q99
    assert only(labelled).magnitude > 1e4 * CHI2_3_Q99


def test_a_flight_row_without_sigma_abstains_as_unknown_uncertainty() -> None:
    case = sc.with_sigma(sc.parabola(), position=0.002)
    result = only(sc.with_values(case, "position_sigma", "ball", [10], np.nan))
    assert result.abstain_reason is AbstainReason.UNKNOWN_UNCERTAINTY


def test_an_unknown_sigma_on_a_flight_too_short_to_score_is_not_the_reason() -> None:
    case = sc.with_sigma(sc.parabola(), position=0.002)
    short = WhitenedFallShapeParams(min_flight_s=100.0)
    result = only(sc.with_values(case, "position_sigma", "ball", [10], np.nan), short)
    assert result.abstain_reason is AbstainReason.TOO_FEW_FRAMES


def test_an_unknown_sigma_in_one_flight_leaves_the_other_flights_candidates() -> None:
    bounce = sc.with_sigma(sc.bounce(e=0.7), position=0.003)
    assert only(bounce).details["mode"] == "consecutive"
    result = only(sc.with_values(bounce, "position_sigma", "ball", [28], np.nan))
    assert math.isfinite(result.magnitude)
    assert result.details["mode"] == "split"
    assert max(hi for _, hi in result.details["windows"]) < sc.EVENT_FRAME


def test_a_pair_adds_the_variances_of_two_objects_with_different_sigma() -> None:
    case = sc.with_sigma(sc.two_balls(1.2 * G), position=0.002)
    equal = run(case)[("ball_i", "ball_j")].magnitude
    wider = sc.with_values(case, "position_sigma", "ball_j", range(sc.FRAMES), 0.004)
    assert run(wider)[("ball_i", "ball_j")].magnitude == pytest.approx(equal * 2.0 / 5.0, rel=1e-9)


def test_the_pair_of_balls_compares_their_first_flights() -> None:
    results = run(sc.with_sigma(sc.two_balls(1.2 * G), position=0.002))
    assert set(results) == {("ball_i",), ("ball_j",), ("ball_i", "ball_j")}
    assert results[("ball_i", "ball_j")].magnitude > 100 * CHI2_3_Q99
    assert results[("ball_i",)].magnitude == pytest.approx(0.0, abs=1e-6)
