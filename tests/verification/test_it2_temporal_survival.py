"""it2 T15 (judgement J5): the injected violations survive LIFT version 3's running median.

Analytic scenes (tests/checks/check_scenarios.py) seen by a camera tilted 25 degrees down, so
vertical motion moves the depth. B is applied to the camera-frame centres and extents of every
visible run exactly as LIFT applies it to its summaries (``stabilize_run``); velocities and
accelerations are then recomputed with the chain's Savitzky-Golay for both tables, and each
check runs on both. (i) R4_size, R4_jump, R4_missing, R1 and R2 must not move; (ii) R3a, R3b,
R5a and R5b must stay closer to their own level than to the level below (0 below L1), and the
clean scene closer to clean than to L1.

Both edge rules are run. With ``"shifted"`` (the plan's rule) the ends of every run lose their
slope: a 3-row r4_resize at a run end loses 9-53 % of its magnitude, so (i) fails and B is not
activated in it2 (J5); a clean parabola's R5a also goes from 0 to about 1.1, which the level
ladder of (ii) does not see. ``"keep"`` passes (i) and (ii); its clean R5a drift (apex clipping,
about 0.03) is reported by the last test. These cases assert the measured failure.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks._common import CommonParams
from physverify.checks.r1_interpenetration import InterpenetrationCheck, InterpenetrationParams
from physverify.checks.r2_equilibrium import EquilibriumCheck, EquilibriumParams
from physverify.checks.r3_collision_energy import (
    EnergyGainCheck,
    EnergyGainParams,
    RestitutionCheck,
    RestitutionParams,
)
from physverify.checks.r4_integrity import (
    ContinuityCheck,
    PermanenceCheck,
    PermanenceParams,
    SizeConstancyCheck,
    SizeParams,
)
from physverify.checks.r5a_fall_shape import FallShapeCheck, FallShapeParams
from physverify.checks.r5b_gravity_value import GravityValueCheck, GravityValueParams
from physverify.extraction.providers._temporal import EdgeMode, stabilize_run
from physverify.extraction.providers.kinematics_savgol import Params as SavgolParams
from physverify.extraction.providers.kinematics_savgol import track_derivatives
from physverify.extraction.providers.lift_backproject import Params as LiftParams
from physverify.physics.geometry import matrix_from_quaternion, quaternion_from_matrix
from physverify.physics.kinematics import SavgolSpec, visible_runs
from physverify.schema.states import ObjectStateTable
from physverify.synth.perturbations.r3_energy_gain import Params as EnergyLevels
from physverify.synth.perturbations.r3_restitution_gain import Params as RestitutionLevels
from physverify.synth.perturbations.r4_resize import Params as ResizeLevels
from physverify.synth.perturbations.r4_teleport import Params as TeleportLevels
from physverify.synth.perturbations.r4_vanish import Params as VanishLevels
from physverify.synth.perturbations.r5a_gravity_switch import Params as SwitchLevels
from physverify.synth.perturbations.r5b_gravity_scale import Params as ScaleLevels

PITCH_DEG = 25.0
REL = 1e-9
WINDOW = LiftParams().temporal_window
SAVGOL = SavgolParams()
Make = Callable[[float], sc.Case]


EDGES = ["shifted", "keep"]


def stabilized(states: ObjectStateTable, edges: EdgeMode) -> ObjectStateTable:
    """LIFT version 3 on a camera-frame table (centre = position, its extents = size)."""
    position, size = states.position.copy(), states.size.copy()
    for n in range(states.n_objects):
        for run in visible_runs(states.visible[:, n]):
            if run.stop - run.start >= WINDOW:
                position[run, n], size[run, n] = stabilize_run(
                    position[run, n], size[run, n], WINDOW, edges
                )
    return replace(states, position=position, size=size)


def kinematics(states: ObjectStateTable) -> ObjectStateTable:
    return track_derivatives(states, SavgolSpec(SAVGOL.window, SAVGOL.polyorder))


def magnitudes(check: Any, case: sc.Case, edges: EdgeMode | None) -> list[float]:
    """Magnitudes without B (``edges`` None) or with B and that edge rule."""
    states = case.states if edges is None else stabilized(case.states, edges)
    inputs = replace(case.inputs(), states=kinematics(states))
    return [r.magnitude for r in check.evaluate(inputs)]


def both(check: Any, case: sc.Case, edges: EdgeMode) -> tuple[list[float], list[float]]:
    return magnitudes(check, case, None), magnitudes(check, case, edges)


def turned(case: sc.Case, yaw_deg: float) -> sc.Case:
    """The world scene turned about the vertical through the world origin, same camera."""
    view, yaw = case.view, sc.yaw(yaw_deg)
    conj = view.rotation @ yaw @ view.rotation.T

    def point(p: np.ndarray) -> np.ndarray:
        world = p @ view.rotation + view.origin
        return np.asarray(view.point(world @ yaw.T))

    s = case.states
    quaternions = np.apply_along_axis(
        lambda q: quaternion_from_matrix(conj @ matrix_from_quaternion(q)), -1, s.quaternion
    )
    states = replace(
        s,
        position=point(s.position),
        velocity=s.velocity @ conj.T,
        acceleration=s.acceleration @ conj.T,
        quaternion=quaternions,
    )
    assert case.contacts is not None
    records = tuple(
        replace(
            r,
            normal=conj @ r.normal,
            points=point(r.points),
            point_normals=None if r.point_normals is None else r.point_normals @ conj.T,
        )
        for r in case.contacts.records
    )
    return replace(case, states=states, contacts=replace(case.contacts, records=records))


def resized(level: float, width: int, where: str) -> sc.Case:
    """r4_resize: the flying ball (1 + level) times larger on ``width`` rows."""
    case = sc.parabola(pitch_deg=PITCH_DEG)
    frames = case.states.t.size
    start = {"start": 0, "middle": 12, "end": frames - width}[where]
    grown = case.states.size[start : start + width, 0] * (1.0 + level)
    return sc.with_values(case, "size", "ball", range(start, start + width), grown)


# --- (i) magnitudes that must not move ---------------------------------------------------------


@pytest.mark.parametrize("where", ["start", "middle", "end"])
@pytest.mark.parametrize("width", [3, 8])
@pytest.mark.parametrize("level", ResizeLevels().levels)
@pytest.mark.parametrize("edges", EDGES)
def test_t15_r4_resize_is_unchanged(edges: EdgeMode, level: float, width: int, where: str) -> None:
    plain, filtered = both(SizeConstancyCheck(SizeParams()), resized(level, width, where), edges)
    print(f"R4_size {edges} m={level} w={width} {where}: {plain} -> {filtered}")
    assert plain[0] == pytest.approx(level, rel=1e-6)
    if edges == "shifted" and width == 3 and where != "middle":
        # The shifted window mixes the plateau with rows whose depth differs by a few percent.
        print(f"  J5 (i) fails: {filtered[0] / plain[0] - 1:+.1%}")
        assert filtered[0] != pytest.approx(plain[0], rel=1e-3)
        return
    assert filtered == pytest.approx(plain, rel=REL)


@pytest.mark.parametrize("level", TeleportLevels().levels)
@pytest.mark.parametrize("edges", EDGES)
def test_t15_r4_teleport_is_unchanged(edges: EdgeMode, level: float) -> None:
    case = sc.teleport(delta=level * 2 * sc.BALL_RADIUS, pitch_deg=PITCH_DEG)
    plain, filtered = both(ContinuityCheck(CommonParams()), case, edges)
    print(f"R4_jump {edges} m={level}: {plain} -> {filtered}")
    assert filtered == pytest.approx(plain, rel=REL)


@pytest.mark.parametrize("rows", [int(n) for n in VanishLevels().levels])
@pytest.mark.parametrize("edges", EDGES)
def test_t15_r4_vanish_is_unchanged(edges: EdgeMode, rows: int) -> None:
    case = sc.disappearance(first=10, last=10 + rows - 1, pitch_deg=PITCH_DEG)
    plain, filtered = both(PermanenceCheck(PermanenceParams()), case, edges)
    assert plain == [rows] and filtered == plain


@pytest.mark.parametrize("depth", [0.01, 0.02, 0.04, 0.08])
@pytest.mark.parametrize("edges", EDGES)
def test_t15_r1_constant_penetration_is_unchanged(edges: EdgeMode, depth: float) -> None:
    for case in (
        sc.overlapping_spheres(depth, pitch_deg=PITCH_DEG),
        sc.box_sphere(depth, pitch_deg=PITCH_DEG),
    ):
        plain, filtered = both(InterpenetrationCheck(InterpenetrationParams()), case, edges)
        assert filtered == pytest.approx(plain, rel=REL)


@pytest.mark.parametrize("offset", [0.21, 0.24, 0.28])
@pytest.mark.parametrize("edges", EDGES)
def test_t15_r2_constant_overhang_is_unchanged(edges: EdgeMode, offset: float) -> None:
    plain, filtered = both(
        EquilibriumCheck(EquilibriumParams()), sc.overhang(offset, pitch_deg=PITCH_DEG), edges
    )
    assert filtered == pytest.approx(plain, rel=REL)


# --- (ii) magnitudes that must stay at their level -----------------------------------------------


def ladder(check: Any, make: Make, levels: tuple[float, ...], label: str, edges: EdgeMode) -> None:
    """J5 (ii): each level L1..L4 stays closer to its own magnitude than to the one below."""
    plain = [magnitudes(check, make(m), None)[0] for m in levels]
    filtered = [magnitudes(check, make(m), edges)[0] for m in levels]
    below = [0.0, *plain[:-1]]
    for level, p, f, lower in zip(levels, plain, filtered, below, strict=True):
        print(f"{label} {edges} m={level}: {p:.5f} -> {f:.5f}")
        assert abs(f - p) < abs(f - lower), (label, level, p, f, lower)


@pytest.mark.parametrize("edges", EDGES)
def test_t15_r3a_restitution_levels_survive(
    edges: EdgeMode,
) -> None:
    def make(m: float) -> sc.Case:
        return sc.bounce(e=1.0 + m, speed=3.0, pitch_deg=PITCH_DEG)

    ladder(RestitutionCheck(RestitutionParams()), make, RestitutionLevels().levels, "R3a", edges)


@pytest.mark.parametrize("yaw_deg", [0.0, 60.0])
@pytest.mark.parametrize("edges", EDGES)
def test_t15_r3b_energy_levels_survive(edges: EdgeMode, yaw_deg: float) -> None:
    def make(m: float) -> sc.Case:
        return turned(sc.collision(gain=m, pitch_deg=PITCH_DEG), yaw_deg)

    ladder(
        EnergyGainCheck(EnergyGainParams()),
        make,
        EnergyLevels().levels,
        f"R3b yaw {yaw_deg}",
        edges,
    )


@pytest.mark.parametrize("edges", EDGES)
def test_t15_r5a_gravity_change_levels_survive(
    edges: EdgeMode,
) -> None:
    def make(m: float) -> sc.Case:
        g = sc.STANDARD_GRAVITY
        return sc.parabola(g=g, g_after=g * (1.0 + m), pitch_deg=PITCH_DEG)

    ladder(FallShapeCheck(FallShapeParams()), make, SwitchLevels().levels, "R5a", edges)


CLEAN_DRIFT = {"shifted": (1.0, 1.2), "keep": (0.02, 0.04)}


@pytest.mark.parametrize("edges", EDGES)
def test_t15_the_clean_parabola_drift_is_the_apex_and_the_ends(edges: EdgeMode) -> None:
    """Not part of J5, reported: on a clean flight the median clips the depth apex (keep) and
    also flattens both ends (shifted); R5a of the clean parabola is 0 without B."""
    case = sc.parabola(pitch_deg=PITCH_DEG)
    check = FallShapeCheck(FallShapeParams())
    assert magnitudes(check, case, None)[0] == pytest.approx(0.0, abs=1e-9)
    low, high = CLEAN_DRIFT[edges]
    assert low < magnitudes(check, case, edges)[0] < high


@pytest.mark.parametrize("edges", EDGES)
def test_t15_r5b_gravity_scale_levels_survive(
    edges: EdgeMode,
) -> None:
    def make(m: float) -> sc.Case:
        return sc.parabola(g=sc.STANDARD_GRAVITY * (1.0 + m), pitch_deg=PITCH_DEG)

    ladder(GravityValueCheck(GravityValueParams()), make, ScaleLevels().levels, "R5b", edges)


def test_t15_the_scenes_move_in_depth() -> None:
    case = sc.parabola(pitch_deg=PITCH_DEG)
    z = case.states.position[:, 0, 2]
    assert float(np.ptp(z)) > 0.1 and not math.isclose(z[0], z[-1])
