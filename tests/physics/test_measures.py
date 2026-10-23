from __future__ import annotations

import math

import numpy as np
import pytest
from difference_sigma import central_difference_sigma, second_difference_sigma

from physverify.physics.finite_diff import (
    central_difference,
    second_difference,
    third_difference_sigma,
)
from physverify.physics.measures.collision_energy import (
    kinetic_energy_gain,
    momentum_mass_ratio,
    restitution,
)
from physverify.physics.measures.gravity_fit import (
    accel_inconsistency,
    fall_fit,
    fall_residual_ratio,
    gravity_rel_error,
    lateral_accel_ratio,
)
from physverify.physics.measures.object_integrity import (
    jump_ratio,
    missing_frames,
    size_change_ratio,
)
from physverify.physics.measures.penetration import penetration_ratio
from physverify.physics.measures.support_overhang import overhang_fraction
from physverify.schema.constants import STANDARD_GRAVITY

DOWN = np.array([0.0, 0.0, -1.0])
SQUARE = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]])


def test_penetration_ratio() -> None:
    assert penetration_ratio(-0.02, (0.1, 0.1, 0.1), (1.0, 1.0, 1.0)) == pytest.approx(0.2)
    assert penetration_ratio(0.5, (1, 1, 1), (1, 1, 1)) == 0.0
    assert math.isnan(penetration_ratio(math.nan, (1, 1, 1), (1, 1, 1)))


@pytest.mark.parametrize(
    ("com", "expected"), [((1.0, 1.0, 5.0), -1.0), ((3.0, 1.0, 0.0), 1.0), ((1.0, -0.5, 0.0), 0.5)]
)
def test_overhang_fraction(com: tuple[float, float, float], expected: float) -> None:
    assert overhang_fraction(com, SQUARE, DOWN, (1.0, 1.0, 1.0)) == pytest.approx(expected)


def test_overhang_on_a_segment_is_never_inside() -> None:
    edge = SQUARE[:2]
    assert overhang_fraction((1.0, 0.0, 0.0), edge, DOWN, (1.0, 1.0, 1.0)) == pytest.approx(0.0)


def test_restitution() -> None:
    assert restitution((0, 0, -2.0), (0, 0, 1.0), (0, 0, 1.0)) == pytest.approx(0.5)


def test_momentum_mass_ratio() -> None:
    ratio, cosine = momentum_mass_ratio((1.0, 0.0, 0.0), (-2.0, 0.0, 0.0))
    assert ratio == pytest.approx(2.0) and cosine == pytest.approx(1.0)


def test_elastic_collision_has_no_energy_gain() -> None:
    gain = kinetic_energy_gain(1.0, (1.0, 0, 0), (0, 0, 0), (0, 0, 0), (1.0, 0, 0))
    assert gain == pytest.approx(0.0)
    assert kinetic_energy_gain(1.0, (1.0, 0, 0), (0, 0, 0), (0, 0, 0), (2.0, 0, 0)) == 3.0


def test_missing_frames() -> None:
    visible = [True, False, False, False, True]
    in_frame = [True, True, True, False, True]
    occlusion = [0.0, 0.95, math.nan, 0.0, 0.0]
    assert missing_frames(visible, in_frame, occlusion) == 1


def test_size_change_ratio() -> None:
    extents = np.array([[1.0, 1.0, 1.0]] * 4 + [[2.0, 2.0, 2.0]] + [[math.nan] * 3])
    assert size_change_ratio(extents) == pytest.approx(1.0)


def test_jump_ratio_and_exclusion() -> None:
    """A one-row spike of 0.5 enters the third difference with weights 1, 3, 3, 1 at rows
    k = 1..4 (of which rows 1..3 have a full stencil): the peak is 3 * 0.5 / (2 L) = 1.5 at
    rows 2 and 3; excluding them leaves row 1 with 0.5 / (2 L) = 0.5."""
    t = np.arange(6, dtype=float)
    positions = np.column_stack([t, np.zeros(6), np.zeros(6)])
    positions[3, 0] += 0.5
    assert jump_ratio(positions, (0.5, 0.5, 0.5)) == pytest.approx(1.5)
    exclude = np.zeros(6, dtype=bool)
    exclude[2:4] = True
    assert jump_ratio(positions, (0.5, 0.5, 0.5), exclude) == pytest.approx(0.5)
    exclude[1] = True
    assert math.isnan(jump_ratio(positions, (0.5, 0.5, 0.5), exclude))


EXTENTS = (0.2, 0.1, 0.4)
LENGTH = (0.2 * 0.1 * 0.4) ** (1.0 / 3.0)


def _parabola(acceleration: tuple[float, float, float], n: int = 24) -> np.ndarray:
    t = np.arange(n, dtype=float) / 24.0
    p0, v0 = np.array([0.1, -0.3, 1.2]), np.array([1.5, 0.4, 2.0])
    return np.asarray(p0 + np.outer(t, v0) + 0.5 * np.outer(t**2, acceleration))


@pytest.mark.parametrize(
    "acceleration",
    [(0.0, 0.0, -9.81), (0.0, 0.0, -2.0), (3.0, -1.0, -14.0), (0.0, 0.0, 0.0), (-4.9, 0.0, 0.0)],
    ids=["g", "weak-g", "tilted-g", "uniform-motion", "uniform-friction"],
)
def test_jump_ratio_vanishes_under_constant_acceleration(
    acceleration: tuple[float, float, float],
) -> None:
    assert jump_ratio(_parabola(acceleration), EXTENTS) < 1e-9


def test_jump_ratio_of_a_body_at_rest_is_zero() -> None:
    assert jump_ratio(np.tile([0.3, 0.2, 0.05], (10, 1)), EXTENTS) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("flight", [True, False])
def test_jump_ratio_of_a_step_is_the_step_over_the_length(flight: bool) -> None:
    rng = np.random.default_rng(3)
    direction = rng.normal(size=3)
    step = 0.07 * direction / np.linalg.norm(direction)
    positions = _parabola((0.0, 0.0, -9.81)) if flight else np.tile([0.3, 0.2, 0.05], (24, 1))
    positions[12:] += step
    assert jump_ratio(positions, EXTENTS) == pytest.approx(0.07 / LENGTH, rel=1e-9)


def test_an_excluded_row_skips_its_four_row_stencil() -> None:
    positions = np.tile([0.3, 0.2, 0.05], (10, 1))
    positions[5:] += [0.0, 0.05, 0.0]
    exclude = np.zeros(10, dtype=bool)
    exclude[3:6] = True
    assert jump_ratio(positions, EXTENTS, exclude) == pytest.approx(0.0, abs=1e-12)
    exclude[5] = False
    assert jump_ratio(positions, EXTENTS, exclude) == pytest.approx(0.025 / LENGTH, rel=1e-9)


def test_jump_ratio_needs_four_rows() -> None:
    assert math.isnan(jump_ratio(np.zeros((3, 3)), EXTENTS))
    assert jump_ratio(np.zeros((4, 3)), EXTENTS) == 0.0


def test_fall_fit_is_exact_on_a_parabola() -> None:
    t = np.linspace(0.2, 1.0, 9)
    a, v0, p0 = np.array([0.1, 0.0, -9.81]), np.array([1.0, 0.5, 0.0]), np.array([0, 0, 3.0])
    tau = t - t[0]
    positions = p0 + np.outer(tau, v0) + 0.5 * np.outer(tau**2, a)
    positions[4] = np.nan
    fit = fall_fit(t, positions)
    assert np.abs(fit.a - a).max() < 1e-9 and np.abs(fit.v0 - v0).max() < 1e-9
    assert fit.n == 8 and fit.rms < 1e-9
    assert fall_residual_ratio(fit, 0.8) < 1e-9


def test_accel_inconsistency_and_lateral_ratio() -> None:
    assert accel_inconsistency([(0, 0, -10.0), (0, 0, -10.0)]) == 0.0
    assert accel_inconsistency([(0, 0, -8.0), (0, 0, -10.0), (0, 0, -12.0)]) == pytest.approx(0.4)
    assert lateral_accel_ratio((3.0, 0.0, -4.0), DOWN) == pytest.approx(0.6)


@pytest.mark.parametrize("k", [0.5, 1.0, 1.25, 2.0])
def test_gravity_rel_error_is_k_minus_one(k: float) -> None:
    meters_per_unit = 0.2
    accel_in_units = k * STANDARD_GRAVITY / meters_per_unit
    assert gravity_rel_error(accel_in_units, meters_per_unit) == pytest.approx(abs(k - 1.0))


def test_differences_are_exact_for_quadratics_on_uneven_grids() -> None:
    t = np.array([0.0, 0.1, 0.25, 0.3, 0.5, 0.8])
    x = np.column_stack([2.0 + 3.0 * t - 4.9 * t**2, t**2])
    velocity = central_difference(x, t)
    acceleration = second_difference(x, t)
    assert np.isnan(velocity[0]).all() and np.isnan(acceleration[-1]).all()
    np.testing.assert_allclose(velocity[1:-1, 0], 3.0 - 9.8 * t[1:-1], atol=1e-12)
    np.testing.assert_allclose(acceleration[1:-1], [[-9.8, 2.0]] * 4, atol=1e-9)


def test_difference_sigmas() -> None:
    assert central_difference_sigma(0.1, 0.5) == pytest.approx(0.1 / (math.sqrt(2) * 0.5))
    assert second_difference_sigma(0.1, 0.5) == pytest.approx(math.sqrt(6) * 0.1 / 0.25)
    assert third_difference_sigma(0.1, 0.5) == pytest.approx(math.sqrt(20) * 0.1 / 0.125)


def _second_difference_coverage(seed: int, replicas: int) -> float:
    """Fraction of noisy replicas whose central second difference lies within one sigma of -g."""
    rng = np.random.default_rng(seed)
    dt, sigma = 1.0 / 24.0, 0.002
    t = np.arange(9) * dt
    truth = 1.0 - 0.5 * STANDARD_GRAVITY * t**2
    noisy = truth[:, None] + rng.normal(0.0, sigma, (t.shape[0], replicas))
    estimates = second_difference(noisy, t)[4]
    band = second_difference_sigma(sigma, dt)
    return float(np.mean(np.abs(estimates + STANDARD_GRAVITY) <= band))


def test_second_difference_sigma_coverage() -> None:
    assert abs(_second_difference_coverage(seed=1, replicas=200) - 0.6827) <= 0.05


def test_second_difference_sigma_is_calibrated_on_many_replicas() -> None:
    assert abs(_second_difference_coverage(seed=2, replicas=100_000) - 0.6827) <= 0.01
