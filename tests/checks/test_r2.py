import math
from dataclasses import replace
from typing import Any

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks import r2_static_equilibrium
from physverify.checks._base import Check
from physverify.checks._params import CommonParams
from physverify.checks.r2_static_equilibrium import PLUGINS, EquilibriumCheck, EquilibriumParams
from physverify.physics.equilibrium import EquilibriumSolution
from physverify.schema.checks import AbstainReason, CheckResult, Verdict
from physverify.schema.contacts import ManifoldKind

NORM_FLOOR = math.cos(math.pi / 8)
# m_min (0.2 / 2 bodies) times the lever arm beyond the base edge over L of the top cube (0.2).
OVERHANG_RESIDUAL = 0.1 * (0.25 - 0.2) / 0.2


def run(case: sc.Case) -> list[CheckResult]:
    return list(EquilibriumCheck(EquilibriumParams()).evaluate(case.inputs()))


def run_one(case: sc.Case) -> CheckResult:
    results = run(case)
    assert len(results) == 1
    return results[0]


def with_states(case: sc.Case, **changes: Any) -> sc.Case:
    return replace(case, states=replace(case.states, **changes))


def point_contact(case: sc.Case) -> sc.Case:
    """The top-on-base record reduced to a single point of unknown extent."""
    assert case.contacts is not None
    records = tuple(
        replace(
            r,
            manifold=ManifoldKind.POINT,
            points=r.points[:1],
            point_normals=None,
            point_distances=None,
        )
        if r.object_a == "top"
        else r
        for r in case.contacts.records
    )
    return replace(case, contacts=replace(case.contacts, records=records))


def test_plugin_contract() -> None:
    assert (EquilibriumCheck,) == PLUGINS
    check = EquilibriumCheck(CommonParams(seed=5))
    assert isinstance(check, Check)
    assert check.params.seed == 5 and check.params.mc_draws == 64


@pytest.mark.parametrize("pitch", [0.0, 30.0])
def test_supported_stack_is_in_equilibrium(pitch: float) -> None:
    result = run_one(sc.stack(pitch_deg=pitch))
    assert result.verdict is Verdict.UNSCORED
    assert result.subject == ("base", "top")
    assert result.magnitude is not None and result.magnitude <= 1e-7
    assert result.sigma == 0.0
    assert result.frame_span == (0, sc.FRAMES - 1)
    assert result.details["masses"]["base"] + result.details["masses"]["top"] == pytest.approx(1)
    assert result.details["excluded_contacts"] == []


@pytest.mark.parametrize("pitch", [0.0, 30.0])
def test_floating_box_is_fully_unsupported(pitch: float) -> None:
    result = run_one(sc.floating_box(pitch_deg=pitch))
    assert result.subject == ("box",)
    assert result.magnitude == pytest.approx(1.0, abs=1e-7)


def test_overhang_with_the_centre_of_mass_inside_balances() -> None:
    result = run_one(sc.overhang(0.15))
    assert result.magnitude is not None and result.magnitude <= 1e-7
    assert result.details["overhang"]["top"] < 0


@pytest.mark.parametrize("yaw", [0.0, 30.0, 45.0])
def test_overhang_with_the_centre_of_mass_outside_is_the_lever_residual(yaw: float) -> None:
    result = run_one(sc.overhang(0.25, yaw_deg=yaw, pitch_deg=30.0))
    assert result.magnitude is not None
    ratio = result.magnitude / OVERHANG_RESIDUAL
    assert NORM_FLOOR - 1e-9 <= ratio <= 1.0 + 1e-9
    assert result.details["masses"]["top"] == pytest.approx(0.1, abs=1e-9)
    assert result.details["overhang"]["top"] == pytest.approx(0.05 / 0.2, abs=1e-9)
    assert result.details["horizontal_norm_bias"] == pytest.approx(1 / NORM_FLOOR - 1)


def test_moving_bodies_are_not_evaluated() -> None:
    assert run(sc.bounce()) == []


def test_point_manifold_abstains() -> None:
    result = run_one(point_contact(sc.stack()))
    assert result.abstain_reason is AbstainReason.CONTACT_GEOMETRY_UNKNOWN


def test_agent_in_contact_is_out_of_scope() -> None:
    case = with_states(sc.stack(), labels=np.array(["box", "hand"]))
    result = run_one(case)
    assert result.abstain_reason is AbstainReason.OUT_OF_SCOPE
    assert result.details["agents"] == ["hand"]


def test_unknown_gravity_abstains_with_missing_input() -> None:
    case = with_states(sc.stack(), gravity_dir=np.full(3, math.nan), gravity_dir_sigma=math.nan)
    result = run_one(case)
    assert result.abstain_reason is AbstainReason.MISSING_INPUT
    assert result.details["missing"] == "gravity_dir"


def test_failed_program_abstains_ill_conditioned(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*_: object) -> EquilibriumSolution:
        nan = np.full(2, math.nan)
        return EquilibriumSolution(math.nan, nan, nan, 4, "numerical difficulties")

    monkeypatch.setattr(r2_static_equilibrium, "solve_equilibrium", failing)
    result = run_one(sc.stack())
    assert result.abstain_reason is AbstainReason.ILL_CONDITIONED
    assert result.details["lp_status"] == 4


@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("offset", [0.0, 0.25])
def test_strided_tables_keep_the_magnitude(quantize: bool, offset: float) -> None:
    case = sc.overhang(offset)
    native = run_one(case)
    strided = run_one(sc.strided(case, 2, quantize=quantize))
    assert native.magnitude is not None and strided.magnitude is not None
    assert strided.magnitude == pytest.approx(native.magnitude, abs=1e-7)
    assert strided.frame_span == (0, 30)
    assert strided.details["rest_rows"] == [0, 15]
    assert strided.details["row_step"] == 2


def test_sigma_is_monte_carlo_over_positions() -> None:
    noisy = run_one(sc.with_sigma(sc.overhang(0.25), position=0.002))
    assert math.isfinite(noisy.sigma) and noisy.sigma > 0
    again = run_one(sc.with_sigma(sc.overhang(0.25), position=0.002))
    assert again.sigma == noisy.sigma
    unknown = run_one(sc.with_sigma(sc.overhang(0.25), position=math.nan))
    assert math.isnan(unknown.sigma)
