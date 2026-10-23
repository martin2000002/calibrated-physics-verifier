"""The injected violations survive ``contacts/kinematic``.

Analytic scenes (tests/checks/check_scenarios.py) seen by a camera tilted 25 degrees down; each
check runs on the scene's states with three contact tables: the scene's true records, those of
``contacts/proximity`` and those of ``contacts/kinematic`` (test camera of
tests/extraction/contact_scenes.py).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

EXTRACTION = str(Path(__file__).resolve().parents[1] / "extraction")
if EXTRACTION not in sys.path:
    sys.path.insert(0, EXTRACTION)

import contact_scenes as cs  # noqa: E402
from contacts_golden_case import proximity_table  # noqa: E402
from reference.r4_jump_v1 import ContinuityCheck  # noqa: E402

from physverify.checks._params import CommonParams  # noqa: E402
from physverify.checks.r1_interpenetration import (  # noqa: E402
    InterpenetrationCheck,
    InterpenetrationParams,
)
from physverify.checks.r2_static_equilibrium import (  # noqa: E402
    EquilibriumCheck,
    EquilibriumParams,
)
from physverify.checks.r3a_bounce_restitution import (  # noqa: E402
    RestitutionCheck,
    RestitutionParams,
)
from physverify.checks.r3b_collision_energy_gain import (  # noqa: E402
    EnergyGainCheck,
    EnergyGainParams,
)
from physverify.checks.r4_missing_permanence import PermanenceCheck, PermanenceParams  # noqa: E402
from physverify.checks.r4_size_constancy import SizeConstancyCheck, SizeParams  # noqa: E402
from physverify.checks.r5a_fall_shape import FallShapeCheck, FallShapeParams  # noqa: E402
from physverify.checks.r5b_gravity_value import (  # noqa: E402
    GravityValueCheck,
    GravityValueParams,
)
from physverify.schema.contacts import ContactTable  # noqa: E402
from physverify.synth.perturbations.r3_energy_gain import Params as EnergyLevels  # noqa: E402
from physverify.synth.perturbations.r3_restitution_gain import (  # noqa: E402
    Params as RestitutionLevels,
)
from physverify.synth.perturbations.r4_teleport import Params as TeleportLevels  # noqa: E402
from physverify.synth.perturbations.r5a_gravity_switch import (  # noqa: E402
    Params as SwitchLevels,
)
from physverify.synth.perturbations.r5b_gravity_scale import Params as ScaleLevels  # noqa: E402

sc = cs.sc
PITCH_DEG = 25.0
REL = 1e-9


def with_contacts(case: sc.Case, contacts: ContactTable) -> sc.Case:
    return replace(case, contacts=contacts)


def proximity(case: sc.Case) -> sc.Case:
    table = proximity_table(case.states)
    assert case.contacts is not None
    return with_contacts(case, replace(case.contacts, records=table.records))


def kinematic(case: sc.Case) -> sc.Case:
    table = cs.kinematic_table(case)
    assert case.contacts is not None
    return with_contacts(case, replace(case.contacts, records=table.records))


def magnitudes(check: Any, case: sc.Case) -> list[float]:
    return [r.magnitude for r in check.evaluate(case.inputs())]


def jump_of_the_ball(case: sc.Case) -> float:
    results = ContinuityCheck(CommonParams()).evaluate(case.inputs())
    (ball,) = [r for r in results if r.subject == ("ball",)]
    return ball.magnitude


# (i) r4_teleport keeps its magnitude.


@pytest.mark.parametrize("flight", [False, True])
@pytest.mark.parametrize("level", TeleportLevels().levels)
def test_t9_i_teleport_jump_equals_the_truth(level: float, flight: bool) -> None:
    case = sc.teleport(delta=level * 2 * sc.BALL_RADIUS, flight=flight, pitch_deg=PITCH_DEG)
    truth, found = jump_of_the_ball(case), jump_of_the_ball(kinematic(case))
    print(f"R4_jump teleport m={level} flight={flight}: truth {truth} kinematic {found}")
    assert found == pytest.approx(truth, rel=REL)
    assert truth == pytest.approx(level, rel=1e-6)


# (ii) a clean bounce no longer looks like a jump.


@pytest.mark.parametrize("beta", [0.1, 0.4, 0.8])
def test_t9_ii_clean_bounce_jump_is_excluded_like_the_truth(beta: float) -> None:
    case = sc.bounce(impact_frame=15.0 + beta, pitch_deg=PITCH_DEG)
    truth = jump_of_the_ball(case)
    near, found = jump_of_the_ball(proximity(case)), jump_of_the_ball(kinematic(case))
    print(f"R4_jump clean bounce {beta}: truth {truth} proximity {near} kinematic {found}")
    assert found <= near
    assert found == pytest.approx(truth, rel=1e-6, abs=1e-12)


# (iii) R3a and R3b keep their levels.


@pytest.mark.parametrize("level", RestitutionLevels().levels)
def test_t9_iii_restitution_gain_matches_the_truth(level: float) -> None:
    case = sc.bounce(e=1.0 + level, impact_frame=15.3, pitch_deg=PITCH_DEG)
    check = RestitutionCheck(RestitutionParams())
    truth, found = magnitudes(check, case), magnitudes(check, kinematic(case))
    print(f"R3a e'=1+{level}: truth {truth} kinematic {found}")
    assert len(found) == len(truth) == 1
    assert abs(found[0] - truth[0]) <= 0.01


@pytest.mark.parametrize("level", EnergyLevels().levels)
def test_t9_iii_energy_gain_matches_the_truth(level: float) -> None:
    case = sc.collision(gain=level, impact_frame=15.3, pitch_deg=PITCH_DEG)
    check = EnergyGainCheck(EnergyGainParams())
    truth, found = magnitudes(check, case), magnitudes(check, kinematic(case))
    print(f"R3b gain {level}: truth {truth} kinematic {found}")
    assert len(found) == len(truth) == 1
    assert abs(found[0] - truth[0]) <= 0.01


# (iv) R5a and R5b: gravity changes are not impacts.


@pytest.mark.parametrize("level", SwitchLevels().levels)
def test_t9_iv_fall_shape_is_unchanged(level: float) -> None:
    g = sc.STANDARD_GRAVITY
    case = sc.parabola(g=g, g_after=g * (1.0 + level), pitch_deg=PITCH_DEG)
    check = FallShapeCheck(FallShapeParams())
    assert magnitudes(check, kinematic(case)) == magnitudes(check, proximity(case))


@pytest.mark.parametrize("level", ScaleLevels().levels)
def test_t9_iv_gravity_value_is_unchanged(level: float) -> None:
    case = sc.parabola(g=sc.STANDARD_GRAVITY * (1.0 + level), pitch_deg=PITCH_DEG)
    check = GravityValueCheck(GravityValueParams())
    assert magnitudes(check, kinematic(case)) == magnitudes(check, proximity(case))


# (v) R1 and R2 on still scenes, R1 on a bounce.


@pytest.mark.parametrize("depth", [0.01, 0.05])
def test_t9_v_penetration_is_unchanged(depth: float) -> None:
    check = InterpenetrationCheck(InterpenetrationParams())
    for case in (
        sc.sinking_box(depth, pitch_deg=PITCH_DEG),
        sc.box_sphere(depth, pitch_deg=PITCH_DEG),
        sc.overlapping_spheres(depth, pitch_deg=PITCH_DEG),
    ):
        assert magnitudes(check, kinematic(case)) == magnitudes(check, proximity(case))


@pytest.mark.parametrize("make", [lambda: sc.stack(pitch_deg=PITCH_DEG), lambda: sc.floating_box()])
def test_t9_v_equilibrium_is_unchanged(make: Any) -> None:
    check = EquilibriumCheck(EquilibriumParams())
    case = make()
    assert magnitudes(check, kinematic(case)) == magnitudes(check, proximity(case))


def test_t9_v_penetration_of_a_bounce_against_the_truth() -> None:
    """The kinematic ENV record is the support plane R1 reads for a ball that never rests;
    its point lies on the ground at the impact, so R1 equals the truth's."""
    case = sc.bounce(impact_frame=15.3, pitch_deg=PITCH_DEG)
    check = InterpenetrationCheck(InterpenetrationParams())
    truth, found = magnitudes(check, case), magnitudes(check, kinematic(case))
    near = magnitudes(check, proximity(case))
    print(f"R1 bounce: truth {truth} proximity {near} kinematic {found}")
    assert found == pytest.approx(truth, abs=1e-9)


# (vi) R4_missing and R4_size do not read contacts.


def test_t9_vi_missing_and_size_are_unchanged() -> None:
    for case, check in (
        (sc.disappearance(pitch_deg=PITCH_DEG), PermanenceCheck(PermanenceParams())),
        (sc.size_change(pitch_deg=PITCH_DEG), SizeConstancyCheck(SizeParams(reference="self"))),
    ):
        assert magnitudes(check, kinematic(case)) == magnitudes(check, proximity(case))
