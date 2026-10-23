import math
from itertools import pairwise

import check_scenarios as sc
import numpy as np
import pytest

from physverify.physics.equilibrium import (
    EquilibriumProblem,
    EquilibriumSettings,
    SupportContact,
    cone_edges,
    horizontal_norm_bias,
    solve_equilibrium,
)
from physverify.physics.geometry import characteristic_length
from physverify.schema.constants import ENV_ID

SETTINGS = EquilibriumSettings(friction=0.6, cone_edges=4, mass_floor=0.2)
NORM_FLOOR = math.cos(math.pi / SETTINGS.norm_edges)
TOP_SIDE = 0.2
BASE_HALF_WIDTH = 0.2


def problem_of(case: sc.Case, row: int = 0) -> EquilibriumProblem:
    """Every body and record of a scene at one row, as the check would pose them."""
    states = case.states
    assert case.contacts is not None
    ids = [str(i) for i in states.object_ids]
    contacts = tuple(
        SupportContact(
            body_a=ids.index(r.object_a),
            body_b=None if r.object_b == ENV_ID else ids.index(r.object_b),
            points=r.points,
            normal=r.normal,
        )
        for r in case.contacts.records
    )
    return EquilibriumProblem(
        centroids=states.position[row],
        lengths=np.array([characteristic_length(s) for s in states.size[row]]),
        gravity_dir=states.gravity_dir,
        contacts=contacts,
    )


def residual(case: sc.Case) -> float:
    solution = solve_equilibrium(problem_of(case), SETTINGS)
    assert solution.status == 0
    return solution.residual


def lever_residual(offset: float) -> float:
    """Minimal torque slack of the overhanging top: m_min times its lever arm over L."""
    m_min = SETTINGS.mass_floor / 2
    return m_min * (offset - BASE_HALF_WIDTH) / TOP_SIDE


@pytest.mark.parametrize("pitch", [0.0, 30.0])
def test_box_on_the_ground_and_a_stack_balance(pitch: float) -> None:
    assert residual(sc.sinking_box(depth=0.0, pitch_deg=pitch)) <= 1e-7
    assert residual(sc.stack(pitch_deg=pitch)) <= 1e-7


@pytest.mark.parametrize("pitch", [0.0, 30.0, 75.0])
def test_floating_body_costs_exactly_its_weight_in_any_camera(pitch: float) -> None:
    solution = solve_equilibrium(problem_of(sc.floating_box(pitch_deg=pitch)), SETTINGS)
    assert solution.residual == pytest.approx(1.0, abs=1e-7)
    assert solution.masses == pytest.approx([1.0], abs=1e-9)
    assert solution.body_residuals == pytest.approx([1.0], abs=1e-7)


def test_floating_body_with_an_arbitrary_gravity_direction() -> None:
    gravity = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    problem = EquilibriumProblem(np.zeros((1, 3)), np.ones(1), gravity, ())
    assert solve_equilibrium(problem, SETTINGS).residual == pytest.approx(1.0, abs=1e-7)


def test_centre_of_mass_inside_the_support_balances() -> None:
    assert residual(sc.overhang(0.15)) <= 1e-7


def test_centre_of_mass_outside_grows_strictly_with_the_overhang() -> None:
    offsets = [0.22, 0.24, 0.26, 0.28]
    values = [residual(sc.overhang(x)) for x in offsets]
    assert values[0] > 1e-4
    assert all(b > a for a, b in pairwise(values))
    for offset, value in zip(offsets, values, strict=True):
        ratio = value / lever_residual(offset)
        assert NORM_FLOOR - 1e-9 <= ratio <= 1.0 + 1e-9


def test_overhanging_body_gets_the_mass_floor() -> None:
    solution = solve_equilibrium(problem_of(sc.overhang(0.25)), SETTINGS)
    base, top = solution.masses
    assert top == pytest.approx(SETTINGS.mass_floor / 2, abs=1e-9)
    assert base + top == pytest.approx(1.0, abs=1e-9)
    assert solution.body_residuals[0] <= 1e-7


@pytest.mark.parametrize("yaw", [0.0, 30.0, 45.0])
@pytest.mark.parametrize("pitch", [0.0, 30.0])
def test_polygonal_norm_bias_stays_within_its_declared_bound(yaw: float, pitch: float) -> None:
    value = residual(sc.overhang(0.25, yaw_deg=yaw, pitch_deg=pitch))
    ratio = value / lever_residual(0.25)
    assert NORM_FLOOR - 1e-9 <= ratio <= 1.0 + 1e-9


def test_horizontal_norm_bias_value() -> None:
    assert horizontal_norm_bias(8) == pytest.approx(1 / math.cos(math.pi / 8) - 1)
    assert horizontal_norm_bias(8) == pytest.approx(0.0824, abs=1e-4)


@pytest.mark.parametrize("normal", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.3, -0.4, 0.5)])
def test_cone_edges_open_at_the_friction_angle(normal: tuple[float, float, float]) -> None:
    friction, n_edges = 0.6, 4
    n = np.asarray(normal) / np.linalg.norm(normal)
    edges = cone_edges(normal, friction, n_edges)
    assert edges.shape == (n_edges, 3)
    unit = edges / np.linalg.norm(edges, axis=1, keepdims=True)
    angles = np.arccos(np.clip(unit @ n, -1.0, 1.0))
    assert angles == pytest.approx(np.full(n_edges, math.atan(friction)), abs=1e-12)
    tangential = edges - np.outer(edges @ n, n)
    assert tangential.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-12)


def test_non_finite_data_fails_without_calling_the_solver() -> None:
    problem = problem_of(sc.stack())
    broken = EquilibriumProblem(
        np.full_like(problem.centroids, math.nan),
        problem.lengths,
        problem.gravity_dir,
        problem.contacts,
    )
    solution = solve_equilibrium(broken, SETTINGS)
    assert solution.status != 0
    assert math.isnan(solution.residual)
    assert np.isnan(solution.masses).all()
