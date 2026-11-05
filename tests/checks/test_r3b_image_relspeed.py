import math
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest
from settings_factory import SettingsFactory

from physverify.app import plugins
from physverify.app.plugins import PLUGIN_KINDS
from physverify.calibration.settings import calibration_settings
from physverify.checks.r3b_collision_energy_gain import EnergyGainCheck
from physverify.checks.variants._image_impulse import apparent_radius, detect
from physverify.checks.variants.r3b_image_relspeed import (
    RelSpeedCheck,
    RelSpeedParams,
    boot_sigma,
)
from physverify.core.errors import ConfigError
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel
from physverify.schema.provenance import GROUND_TRUTH
from physverify.schema.stages import Stage
from physverify.verification._base import POLICIES
from physverify.verification.runner import configured_suite
from physverify.verification.sweep import CheckUnit

TRUTH = {Stage.CONTACTS.value: GROUND_TRUTH, Stage.LIFT.value: GROUND_TRUTH}
PERCEIVED: dict[str, str] = {}
UNIT = "log_normal_speed_ratio"


def scored(case: sc.Case, provenance: dict[str, str]) -> list[CheckResult]:
    inputs = case.inputs(InputLevel.PERCEIVED, provenance)
    return RelSpeedCheck(RelSpeedParams()).evaluate(inputs)


def only(case: sc.Case, provenance: dict[str, str]) -> CheckResult:
    (result,) = scored(case, provenance)
    return result


def scaled_depth(case: sc.Case, seed: int) -> sc.Case:
    """Every centre moved along its pixel ray by a factor per row and body."""
    position = np.asarray(case.states.position, dtype=np.float64)
    factor = np.random.default_rng(seed).uniform(0.5, 2.0, (*position.shape[:2], 1))
    return replace(case, states=replace(case.states, position=position * factor))


@pytest.mark.parametrize("provenance", [TRUTH, PERCEIVED], ids=["truth", "detected"])
@pytest.mark.parametrize("gain", [0.0, 0.2, -0.36])
def test_an_in_plane_impact_reads_the_log_of_its_restitution(
    gain: float, provenance: dict[str, str]
) -> None:
    result = only(sc.collision(gain=gain), provenance)
    assert result.magnitude == pytest.approx(0.5 * math.log1p(gain), abs=1e-12)
    assert result.details["t_e"] == pytest.approx(sc.EVENT_FRAME, abs=1e-9)
    assert result.unit == UNIT


@pytest.mark.parametrize("provenance", [TRUTH, PERCEIVED], ids=["truth", "detected"])
def test_the_ratio_ignores_the_depth_of_each_body(provenance: dict[str, str]) -> None:
    case = sc.collision(gain=0.2, mass_ratio=2.0)
    moved = scaled_depth(case, seed=3)
    assert only(moved, provenance).magnitude == pytest.approx(
        only(case, provenance).magnitude, abs=1e-12
    )


def test_unequal_masses_read_the_relative_restitution() -> None:
    r, gain = 3.0, -0.19
    centre, energy = (r - 1.0) / (r + 1.0), r + 1.0
    after = math.sqrt(((1 + gain) * energy - (r + 1) * centre**2) * (r + 1) / r)
    result = only(sc.collision(gain=gain, mass_ratio=r), TRUTH)
    assert result.magnitude == pytest.approx(math.log(after / 2.0), abs=1e-12)


def test_a_strided_track_reads_the_same_ratio() -> None:
    case = sc.strided(sc.collision(gain=0.2), 2, quantize=False)
    assert only(case, PERCEIVED).magnitude == pytest.approx(0.5 * math.log1p(0.2), abs=1e-12)


def test_a_kink_far_from_contact_is_not_an_event() -> None:
    case = sc.collision(gain=0.2)
    position = np.array(case.states.position)
    position[:, 1, 1] += 1.0
    apart = replace(case, states=replace(case.states, position=position))
    assert detect(apart.states, apart.contacts, RelSpeedParams().gates()) == []
    assert scored(apart, PERCEIVED) == []


def test_the_radius_window_and_the_own_record_tolerance_are_params() -> None:
    params = RelSpeedParams()
    assert (params.gates().radius_rows, params.same_t_frames) == (2, 1e-9)
    assert RelSpeedParams(radius_rows=0).gates().radius_rows == 0
    states = sc.collision(gain=0.2).states
    size, z = np.asarray(states.size[5, 0]), float(states.position[5, 0, 2])
    assert apparent_radius(states, 0, 5, 0) == 0.25 * (size[0] + size[1]) / z


def test_a_noisy_track_is_detected_near_the_impact_with_a_finite_sigma() -> None:
    case = sc.jittered(sc.collision(gain=0.2), 1e-3, seed=5)
    detected, truth = only(case, PERCEIVED), only(case, TRUTH)
    assert abs(detected.details["t_e"] - sc.EVENT_FRAME) < 1.0
    for result in (detected, truth):
        assert result.magnitude == pytest.approx(0.5 * math.log1p(0.2), abs=0.05)
        assert 0.0 < result.sigma < 0.1
    assert only(case, PERCEIVED).sigma == detected.sigma


def test_the_bootstrap_is_unknown_when_most_draws_have_no_ratio() -> None:
    t = [np.arange(-6.0, 0.0), np.arange(1.0, 7.0)]
    rng = np.random.default_rng(0)
    y = [np.column_stack([ti, np.zeros(6)]) + rng.normal(0.0, 1e-3, (6, 2)) for ti in t]
    sigma, finite = boot_sigma(t, y, 0.5, np.random.default_rng(1), RelSpeedParams())
    assert math.isnan(sigma) and finite < 100


def test_the_depth_view_is_reported_only_on_ground_truth() -> None:
    case = sc.collision(gain=0.2)
    truth = only(case, TRUTH)
    assert truth.details["s_3d"] == pytest.approx(truth.magnitude, abs=1e-12)
    assert truth.details["phi_deg"] == pytest.approx(0.0, abs=1e-9)
    assert "s_3d" not in only(case, {Stage.CONTACTS.value: GROUND_TRUTH}).details
    assert "s_3d" not in only(case, PERCEIVED).details


def test_the_variant_abstains_where_the_ratio_is_not_defined() -> None:
    ground = sc.with_records(sc.collision(), sc.ground_impact(15.0, 15.0))
    assert only(ground, TRUTH).abstain_reason is AbstainReason.OUT_OF_SCOPE
    early = sc.collision(impact_frame=1.4)
    assert only(early, TRUTH).abstain_reason is AbstainReason.TOO_FEW_FRAMES
    case = sc.collision()
    assert case.contacts is not None
    moved = tuple(replace(r, t_start=8.0, t_end=8.0) for r in case.contacts.records)
    no_kink = replace(case, contacts=replace(case.contacts, records=moved))
    assert only(no_kink, TRUTH).abstain_reason is AbstainReason.ILL_CONDITIONED
    bare = replace(case, contacts=None)
    assert only(bare, PERCEIVED).abstain_reason is AbstainReason.MISSING_INPUT


def test_it_is_the_selectable_image_relspeed_variant_of_r3b(make_settings: SettingsFactory) -> None:
    chosen = plugins.build("check_variants")["R3b/image_relspeed"]
    assert chosen is RelSpeedCheck
    assert chosen.requires == EnergyGainCheck.requires
    assert CheckUnit.of(chosen).written_unit == EnergyGainCheck.magnitude_unit
    selected = plugins.configured("checks", make_settings("check_variants.R3b=image_relspeed"))
    assert selected["R3b"] is RelSpeedCheck
    for profile in (None, "fast", "precise"):
        assert plugins.configured("checks", make_settings(profile=profile))["R3b"] is (
            EnergyGainCheck
        )


def test_its_tau_and_tolerance_must_be_declared_in_its_unit(make_settings: SettingsFactory) -> None:
    policies = PLUGIN_KINDS[POLICIES.name]()
    chosen = "check_variants.R3b=image_relspeed"
    bare = make_settings(chosen)
    with pytest.raises(ConfigError, match=r"R3b \(tau in dE_over_E, check reports log_normal"):
        configured_suite(bare, plugins.configured("checks", bare), policies)
    with pytest.raises(ConfigError, match=r"tau_tolerance\.R3b is in dE_over_E"):
        calibration_settings(bare).tolerance_of(CheckUnit.of(RelSpeedCheck))
    declared = make_settings(
        chosen,
        "verification.thresholds.R3b=0.0",
        f"verification.threshold_units.R3b={UNIT}",
        f"calibration.tau_tolerance_units.R3b={UNIT}",
    )
    suite = configured_suite(declared, plugins.configured("checks", declared), policies)
    assert suite.thresholds.tau_of("R3b") == 0.0
    assert calibration_settings(declared).tolerance_of(CheckUnit.of(RelSpeedCheck)) == 1e-5
