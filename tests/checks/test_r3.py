import math
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks._base import Check
from physverify.checks._params import CommonParams
from physverify.checks.r3_collision_energy import (
    PLUGINS,
    EnergyGainCheck,
    EnergyGainParams,
    RestitutionCheck,
    RestitutionParams,
)
from physverify.extraction.providers.kinematics_savgol import track_derivatives
from physverify.physics.kinematics import SavgolSpec
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ManifoldKind
from physverify.schema.states import PrimitiveKind

# Native frames: 6 fit rows on each side of the contact rows 15..16 of the impact at 15.4.
NATIVE_SPAN = (9, 22)
# Stride 2 widens the contact rows by q = 1 (frames 14..16); snapping the record to frame 16
# leaves only that row in contact.
STRIDED_SPANS = {False: (2, 28), True: (4, 28)}
SAVGOL = SavgolSpec(window=7, polyorder=2)
POSITION_SIGMA = 0.002
R3A_THRESHOLD = 0.05  # provisional threshold of configs/checks.toml
COVERAGE_SEEDS = 200


def bounce_result(case: sc.Case) -> CheckResult:
    results = RestitutionCheck(RestitutionParams()).evaluate(case.inputs())
    assert [r.subject for r in results] == [("ball", ENV_ID)]
    return results[0]


def collision_result(case: sc.Case) -> CheckResult:
    results = EnergyGainCheck(EnergyGainParams()).evaluate(case.inputs())
    assert [r.subject for r in results] == [("ball_i", "ball_j")]
    return results[0]


def ground_record(t_start: float, t_end: float) -> ContactRecord:
    return ContactRecord(
        object_a="ball_i",
        object_b=ENV_ID,
        t_start=t_start,
        t_end=t_end,
        kind=ContactKind.IMPACT,
        normal=np.array([0.0, 0.0, 1.0]),
        points=np.array([[-0.1, 0.0, 0.9]]),
        manifold=ManifoldKind.POINT,
        confidence=1.0,
        source="test",
    )


def test_plugin_contract() -> None:
    assert (RestitutionCheck, EnergyGainCheck) == PLUGINS
    for cls in PLUGINS:
        check = cls(CommonParams(seed=7))
        assert isinstance(check, Check)
        assert check.params.seed == 7


def test_missing_contacts_abstain() -> None:
    for cls, case in ((RestitutionCheck, sc.bounce()), (EnergyGainCheck, sc.collision())):
        inputs = replace(case.inputs(), contacts=None)
        (result,) = cls(CommonParams()).evaluate(inputs)
        assert result.abstain_reason is AbstainReason.MISSING_INPUT


@pytest.mark.parametrize("pitch", [0.0, 30.0])
@pytest.mark.parametrize("e", [0.8, 1.2])
def test_restitution_is_e_minus_one(e: float, pitch: float) -> None:
    result = bounce_result(sc.bounce(e=e, pitch_deg=pitch))
    assert result.verdict is Verdict.UNSCORED
    assert result.magnitude == pytest.approx(e - 1.0, abs=1e-6)
    assert result.sigma == 0.0
    assert result.frame_span == NATIVE_SPAN
    assert result.details["t_c"] == pytest.approx(sc.EVENT_FRAME * sc.DT, abs=1e-6)
    assert result.details["t_c_method"] == "plane"


def test_restitution_of_a_box_is_out_of_scope() -> None:
    result = bounce_result(sc.bounce(kind=PrimitiveKind.BOX))
    assert result.abstain_reason is AbstainReason.OUT_OF_SCOPE
    assert result.details["primitive"] == "box"


@pytest.mark.parametrize("quantize", [False, True])
def test_strided_restitution_keeps_the_magnitude(quantize: bool) -> None:
    result = bounce_result(sc.strided(sc.bounce(), 2, quantize=quantize))
    assert result.magnitude == pytest.approx(-0.2, abs=1e-6)
    assert result.frame_span == STRIDED_SPANS[quantize]
    assert result.details["row_step"] == 2


def test_impact_near_the_clip_start_has_too_few_frames() -> None:
    result = bounce_result(sc.bounce(impact_frame=1.4))
    assert result.abstain_reason is AbstainReason.TOO_FEW_FRAMES
    assert result.details["rows_before"] < 3


def test_grazing_impact_is_ill_conditioned() -> None:
    result = bounce_result(sc.bounce(speed=0.05))
    assert result.abstain_reason is AbstainReason.ILL_CONDITIONED


def test_restitution_sigma_follows_the_position_sigma() -> None:
    noisy = bounce_result(sc.with_sigma(sc.bounce(), position=0.01))
    assert math.isfinite(noisy.sigma) and noisy.sigma > 0
    unknown = bounce_result(sc.with_sigma(sc.bounce(), position=math.nan))
    assert math.isnan(unknown.sigma)
    legacy = RestitutionCheck(RestitutionParams(fit_source="velocity"))
    (velocity,) = legacy.evaluate(sc.with_sigma(sc.bounce(), velocity=0.01).inputs())
    assert math.isfinite(velocity.sigma) and velocity.sigma > 0


def test_params_need_enough_rows_for_the_fit() -> None:
    with pytest.raises(ValueError):
        RestitutionParams(min_side_frames=2)
    with pytest.raises(ValueError):
        RestitutionParams(fit_frames=3, min_side_frames=4)
    assert RestitutionParams(fit_source="velocity", min_side_frames=2).min_side_frames == 2


def perceived(case: sc.Case, seed: int, shift: float = 0.0) -> sc.Case:
    """Noisy positions with their sigma, velocities from Savitzky-Golay (as A5 emits them) and
    contact records moved by ``shift`` native frames."""
    states = case.states
    rng = np.random.default_rng(seed)
    noisy = replace(
        states,
        position=states.position + rng.normal(0.0, POSITION_SIGMA, states.position.shape),
        position_sigma=np.full(states.position.shape, POSITION_SIGMA),
    )
    assert case.contacts is not None
    records = tuple(
        replace(r, t_start=r.t_start + shift, t_end=r.t_end + shift) for r in case.contacts.records
    )
    contacts = replace(case.contacts, records=records)
    return replace(case, states=track_derivatives(noisy, SAVGOL), contacts=contacts)


def perceived_result(case: sc.Case, params: RestitutionParams | None = None) -> CheckResult:
    check = RestitutionCheck(params or RestitutionParams())
    (result,) = check.evaluate(case.inputs(level=InputLevel.PERCEIVED))
    assert result.magnitude is not None
    return result


@pytest.mark.parametrize("shift", [0.0, -1.0, 1.0])
def test_perceived_elastic_bounce_keeps_its_energy(shift: float) -> None:
    result = perceived_result(perceived(sc.bounce(e=1.0), seed=1, shift=shift))
    assert result.magnitude is not None
    assert abs(result.magnitude) < R3A_THRESHOLD
    assert abs(result.magnitude) <= 3.0 * result.sigma


def test_smoothed_velocities_bias_the_legacy_fit() -> None:
    case = perceived(sc.bounce(e=1.0), seed=1)
    legacy = perceived_result(case, RestitutionParams(fit_source="velocity", min_side_frames=2))
    assert legacy.magnitude is not None
    assert abs(legacy.magnitude) > R3A_THRESHOLD


def test_perceived_inelastic_bounce_is_detected() -> None:
    result = perceived_result(perceived(sc.bounce(e=0.5), seed=2))
    assert result.magnitude == pytest.approx(-0.5, abs=3.0 * result.sigma)
    assert abs(result.magnitude or 0.0) - 2.0 * result.sigma > R3A_THRESHOLD


def test_restitution_sigma_covers_the_error() -> None:
    e = 0.8
    z = []
    for seed in range(COVERAGE_SEEDS):
        # An own MC stream per seed: a shared one would give every seed the same sigma error.
        params = RestitutionParams(seed=seed)
        result = perceived_result(perceived(sc.bounce(e=e), seed=seed), params)
        assert result.magnitude is not None
        z.append((result.magnitude - (e - 1.0)) / result.sigma)
    coverage = float(np.mean(np.abs(z) <= 2.0))
    assert 0.92 <= coverage <= 0.98


@pytest.mark.parametrize("pitch", [0.0, 30.0])
def test_elastic_collision_of_equal_masses_has_no_gain(pitch: float) -> None:
    result = collision_result(sc.collision(pitch_deg=pitch))
    assert result.verdict is Verdict.UNSCORED
    assert result.magnitude == pytest.approx(0.0, abs=1e-6)
    assert result.sigma == 0.0
    assert result.frame_span == NATIVE_SPAN
    assert result.details["mass_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert result.details["mass_ratio_source"] == "momentum"
    assert result.details["rot_energy_fraction"] == 0.0


def test_energy_gain_with_unequal_masses() -> None:
    result = collision_result(sc.collision(mass_ratio=3.0, gain=0.2))
    assert result.magnitude == pytest.approx(0.2, abs=1e-6)
    assert result.details["mass_ratio"] == pytest.approx(3.0, abs=1e-9)
    assert result.details["cos_antiparallel"] == pytest.approx(1.0, abs=1e-9)


def test_skewed_impulses_need_the_prompt_for_the_mass_ratio() -> None:
    skewed = sc.collision(skew_deg=40.0)
    unknown = collision_result(skewed)
    assert unknown.abstain_reason is AbstainReason.ILL_CONDITIONED
    assert unknown.details["cos_antiparallel"] == pytest.approx(math.cos(math.radians(40.0)))
    prompted = collision_result(sc.with_prompt(skewed, "Two identical balls collide"))
    assert prompted.verdict is Verdict.UNSCORED
    assert prompted.details["mass_ratio"] == 1.0
    assert prompted.details["mass_ratio_source"] == "prompt"


def test_ground_contact_during_the_collision_is_out_of_scope() -> None:
    result = collision_result(sc.with_records(sc.collision(), ground_record(15.0, 15.0)))
    assert result.abstain_reason is AbstainReason.OUT_OF_SCOPE
    assert result.details["external_contacts"] == ["ball_i|impact|15-15"]


def test_distant_ground_contact_is_ignored() -> None:
    result = collision_result(sc.with_records(sc.collision(), ground_record(2.0, 3.0)))
    assert result.verdict is Verdict.UNSCORED
    assert result.magnitude == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("quantize", [False, True])
def test_strided_collision_keeps_the_magnitude(quantize: bool) -> None:
    case = sc.collision(mass_ratio=3.0, gain=0.2)
    result = collision_result(sc.strided(case, 2, quantize=quantize))
    assert result.magnitude == pytest.approx(0.2, abs=1e-6)
    assert result.frame_span == STRIDED_SPANS[quantize]
    assert result.details["t_c_quantization_s"] == pytest.approx(sc.DT)


def test_collision_sigma_is_monte_carlo_over_the_fits() -> None:
    noisy = sc.with_sigma(sc.collision(mass_ratio=3.0, gain=0.2), position=0.01)
    first, second = collision_result(noisy), collision_result(noisy)
    assert math.isfinite(first.sigma) and first.sigma > 0
    assert second.sigma == first.sigma
    unknown = sc.with_sigma(sc.collision(), position=math.nan)
    assert math.isnan(collision_result(unknown).sigma)
