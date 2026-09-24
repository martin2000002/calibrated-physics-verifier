"""Closed-form flights used by the trajectory perturbations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.synth.perturbations._ballistics import (
    GravitySchedule,
    energy_scale,
    impact_velocity,
    integrate_sphere,
    propagate_quaternion,
    rolling_angular_velocity,
    time_to_height,
)

G = np.array([0.0, 0.0, -9.81])


def test_time_to_height_matches_free_fall() -> None:
    assert time_to_height(1.0, 0.0, -9.81, 0.0) == pytest.approx(math.sqrt(2.0 / 9.81))
    assert time_to_height(1.0, 3.0, -9.81, 0.0) > time_to_height(1.0, 0.0, -9.81, 0.0)
    assert time_to_height(1.0, 1.0, 0.0, 0.0) == math.inf
    assert time_to_height(0.0, 1.0, -9.81, 2.0) == math.inf


def test_bounces_follow_the_restitution_schedule() -> None:
    times = np.arange(48) / 24.0
    flight = integrate_sphere(
        (0.0, 0.0, 1.0), (0.5, 0.0, 0.0), 0.0, 0.1, GravitySchedule.constant(G), (0.8, 0.5), times
    )
    first, second = flight.bounces[:2]
    assert first.t == pytest.approx(math.sqrt(2 * 0.9 / 9.81))
    assert first.v_after[2] == pytest.approx(-0.8 * first.v_before[2])
    assert second.v_after[2] == pytest.approx(-0.5 * second.v_before[2])
    assert np.all(flight.positions[:, 2] >= 0.1 - 1e-9)
    flying = times <= second.t
    np.testing.assert_allclose(flight.positions[flying, 0], 0.5 * times[flying], atol=1e-9)


def test_gravity_switch_changes_the_acceleration() -> None:
    tilted = np.array([3.0, 0.0, -9.0])
    schedule = GravitySchedule((0.25,), (G, tilted))
    times = np.array([0.0, 0.25, 0.5])
    flight = integrate_sphere((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), 0.0, 0.1, schedule, 1.0, times)
    v = flight.velocities
    np.testing.assert_allclose(v[1], G * 0.25, atol=1e-12)
    np.testing.assert_allclose(v[2], G * 0.25 + tilted * 0.25, atol=1e-12)


def test_impact_velocity_extrapolates_ballistically() -> None:
    np.testing.assert_allclose(impact_velocity((0, 0, -1.0), 0.0, 0.1, G), (0, 0, -1.981))


def test_rolling_and_quaternion_propagation() -> None:
    omega = rolling_angular_velocity((1.0, 0.0, 0.0), 0.5)
    np.testing.assert_allclose(omega, (0.0, 2.0, 0.0))
    q = propagate_quaternion((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, math.pi), 1.0)
    np.testing.assert_allclose(q, (0.0, 0.0, 0.0, 1.0), atol=1e-12)
    assert np.linalg.norm(propagate_quaternion(q, omega, 0.3)) == pytest.approx(1.0)


def test_energy_scale_reaches_the_requested_gain() -> None:
    masses = [1.0, 2.0]
    before = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
    after = [np.array([-0.2, 0.0, 0.0]), np.array([0.6, 0.0, 0.0])]
    lam = energy_scale(masses, before, after, 0.25)
    m = np.asarray(masses)
    cm = sum(mi * v for mi, v in zip(m, after, strict=True)) / m.sum()
    scaled = [cm + lam * (v - cm) for v in after]
    energy = lambda vs: 0.5 * sum(mi * float(v @ v) for mi, v in zip(m, vs, strict=True))  # noqa: E731
    assert energy(scaled) == pytest.approx(1.25 * energy(before))
    assert math.isnan(energy_scale(masses, before, [after[0], after[0]], -1.0))
