"""R2 footprint: each body resting on other bodies against the outline of its supports."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import check_scenarios as sc
import numpy as np
import pytest
from settings_factory import SettingsFactory

from physverify.app import plugins
from physverify.checks._base import CheckInputs
from physverify.checks.r2_static_equilibrium import EquilibriumCheck
from physverify.checks.variants.r2_footprint import (
    SUPPORT_MAX_TILT_DEG,
    FootprintParams,
    SupportFootprintCheck,
)
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel
from physverify.schema.constants import ENV_ID
from physverify.schema.error_profile import SUPPORT_OFFSET_REL, ErrorEntry
from physverify.schema.stages import Stage

PROVIDER = "backproject"
PERCEIVED = {Stage.LIFT.value: PROVIDER, "scale.source": "size_prior"}
TOP_LENGTH = 0.2


def run(inputs: CheckInputs, **params: Any) -> list[CheckResult]:
    return list(SupportFootprintCheck(FootprintParams(**params)).evaluate(inputs))


def run_one(case: sc.Case, **params: Any) -> CheckResult:
    (result,) = run(case.inputs(), **params)
    return result


def with_states(case: sc.Case, **changes: Any) -> sc.Case:
    return replace(case, states=replace(case.states, **changes))


def entry(provider: str = PROVIDER, sigma: float = 0.07) -> ErrorEntry:
    return ErrorEntry(
        stage=Stage.LIFT,
        provider=provider,
        provider_version="1",
        quantity=SUPPORT_OFFSET_REL,
        unit="1",
        bias=0.0,
        sigma=sigma,
        n=40,
    )


def perceived(case: sc.Case, *errors: ErrorEntry) -> CheckInputs:
    return replace(case.inputs(InputLevel.PERCEIVED, PERCEIVED), errors=errors)


def without_ground(case: sc.Case) -> sc.Case:
    assert case.contacts is not None
    records = tuple(r for r in case.contacts.records if r.object_b != ENV_ID)
    return replace(case, contacts=replace(case.contacts, records=records))


def tilted_support(case: sc.Case, degrees: float) -> sc.Case:
    """The top-on-base record with its normal leaned ``degrees`` from the vertical."""
    assert case.contacts is not None
    up = case.view.vector(sc.UP)
    side = case.view.vector((1.0, 0.0, 0.0))
    a = math.radians(degrees)
    normal = math.cos(a) * up + math.sin(a) * side
    records = tuple(
        replace(r, normal=normal) if r.object_a == "top" else r for r in case.contacts.records
    )
    return replace(case, contacts=replace(case.contacts, records=records))


def test_the_plugin_is_the_r2_footprint_variant() -> None:
    assert plugins.build("check_variants")["R2/footprint"] is SupportFootprintCheck
    assert SupportFootprintCheck.magnitude_unit == "overhang_over_size"
    assert SUPPORT_MAX_TILT_DEG == pytest.approx(45.0)


def test_the_setting_selects_it_and_default_restores_v1(make_settings: SettingsFactory) -> None:
    chosen = plugins.configured("checks", make_settings("check_variants.R2=footprint"))
    assert chosen["R2"] is SupportFootprintCheck
    assert plugins.configured("checks", make_settings())["R2"] is EquilibriumCheck
    restored = plugins.configured("checks", make_settings("check_variants.R2=default"))
    assert restored["R2"] is EquilibriumCheck


def test_a_centred_top_is_one_half_base_inside() -> None:
    result = run_one(sc.stack())
    assert result.abstain_reason is None and result.sigma == 0.0
    assert result.magnitude == pytest.approx(-0.2 / TOP_LENGTH, abs=1e-9)
    assert result.details["supports"] == {"top": ["base"]}


@pytest.mark.parametrize(("yaw", "pitch"), [(0.0, 0.0), (30.0, 30.0), (45.0, 10.0)])
def test_the_overhang_is_the_centre_beyond_the_edge_over_the_length(
    yaw: float, pitch: float
) -> None:
    result = run_one(sc.overhang(0.28, yaw_deg=yaw, pitch_deg=pitch))
    assert result.magnitude == pytest.approx(0.08 / TOP_LENGTH, abs=1e-9)


def test_a_cylinder_support_uses_its_rim() -> None:
    case = with_states(sc.overhang(0.28), primitive=np.array(["cylinder", "box"]))
    assert run_one(case).magnitude == pytest.approx(0.08 / TOP_LENGTH, abs=1e-9)


def test_a_body_on_the_floor_alone_gives_no_result() -> None:
    box = sc.still("box", sc.PrimitiveKind.BOX, sc.cube(0.2), (0.0, 0.0, 0.1))
    assert run(sc.build("floor", [box], [sc.resting(box)]).inputs()) == []


def test_moving_bodies_are_not_evaluated() -> None:
    assert run(sc.bounce().inputs()) == []


@pytest.mark.parametrize(("tilt", "results"), [(30.0, 1), (60.0, 0)])
def test_a_contact_steeper_than_the_friction_limit_is_no_support(tilt: float, results: int) -> None:
    assert len(run(tilted_support(sc.stack(), tilt).inputs())) == results


def test_a_sphere_support_has_no_outline() -> None:
    case = with_states(sc.stack(), primitive=np.array(["sphere", "box"]))
    result = run_one(case)
    assert result.abstain_reason is AbstainReason.CONTACT_GEOMETRY_UNKNOWN
    assert result.details["point_support"] == "base"


def test_an_unknown_support_pose_is_a_missing_input() -> None:
    case = sc.with_values(sc.stack(), "quaternion", "base", range(sc.FRAMES), np.nan)
    result = run_one(case)
    assert result.abstain_reason is AbstainReason.MISSING_INPUT
    assert result.details["missing"] == "pose:base"


def test_a_support_the_body_does_not_reach_is_inconsistent() -> None:
    case = sc.stack()
    lifted = case.view.point((0.0, 0.0, 0.3 + TOP_LENGTH))
    case = sc.with_values(case, "position", "top", range(sc.FRAMES), lifted)
    result = run_one(case)
    assert result.abstain_reason is AbstainReason.SUPPORT_INCONSISTENT
    assert result.details["support_gap_over_length"] == pytest.approx(1.0, abs=1e-9)


def test_a_gap_within_half_a_length_is_scored() -> None:
    case = sc.stack()
    lifted = case.view.point((0.0, 0.0, 0.3 + 0.4 * TOP_LENGTH))
    case = sc.with_values(case, "position", "top", range(sc.FRAMES), lifted)
    assert run_one(case).abstain_reason is None


def test_without_contacts_the_check_abstains() -> None:
    (result,) = run(replace(sc.stack().inputs(), contacts=None))
    assert result.abstain_reason is AbstainReason.MISSING_INPUT
    assert result.details["missing"] == "contacts"


def test_unknown_gravity_is_a_missing_input() -> None:
    case = with_states(sc.stack(), gravity_dir=np.full(3, math.nan), gravity_dir_sigma=math.nan)
    assert run_one(case).abstain_reason is AbstainReason.MISSING_INPUT


def test_an_agent_in_the_stack_is_out_of_scope() -> None:
    result = run_one(with_states(sc.stack(), labels=np.array(["box", "hand"])))
    assert result.abstain_reason is AbstainReason.OUT_OF_SCOPE


def test_perceived_bodies_without_a_path_to_the_ground_abstain() -> None:
    (result,) = run(perceived(without_ground(sc.stack()), entry()))
    assert result.abstain_reason is AbstainReason.NO_SUPPORT_PATH


def test_the_oracle_scores_a_stack_without_a_ground_path() -> None:
    assert run_one(without_ground(sc.stack())).abstain_reason is None


def test_perceived_sigma_is_the_measured_offset_error_of_the_lift() -> None:
    (result,) = run(perceived(sc.overhang(0.28), entry(sigma=0.07)))
    assert result.abstain_reason is None
    assert result.sigma == pytest.approx(0.07)
    assert result.magnitude == pytest.approx(0.4, abs=1e-9)
    assert result.details["sigma_source"].startswith(f"lift/{PROVIDER}")


@pytest.mark.parametrize("errors", [(), (entry(provider="ground_anchored"),)])
def test_perceived_without_the_lift_entry_has_unknown_uncertainty(
    errors: tuple[ErrorEntry, ...],
) -> None:
    (result,) = run(perceived(sc.stack(), *errors))
    assert result.abstain_reason is AbstainReason.UNKNOWN_UNCERTAINTY
    assert result.details["missing"] == SUPPORT_OFFSET_REL


def test_v1_params_stay_valid_for_the_variant() -> None:
    params = FootprintParams.model_validate({"mass_floor": 0.3})
    assert params.support_max_tilt_deg == pytest.approx(SUPPORT_MAX_TILT_DEG)
    with pytest.raises(ValueError, match="less than 90"):
        FootprintParams(support_max_tilt_deg=90.0)
