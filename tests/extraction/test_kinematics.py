"""AC8: Savitzky-Golay derivatives, their propagated sigma and run splitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from difference_sigma import central_difference_sigma, second_difference_sigma
from support import make_context, run_provider, sampling_of
from synthetic_clip import SyntheticClip

from physverify.extraction.matching import align_ground_truth
from physverify.extraction.providers.kinematics_savgol import SavgolKinematics
from physverify.physics.kinematics import (
    SavgolSpec,
    differentiate,
    propagate_linear_sigma,
    savgol_operator,
)
from physverify.physics.uncertainty import monte_carlo
from physverify.schema.stages import Stage
from physverify.schema.states import ObjectStateTable

DT = 1.0 / 24.0
N = 12
DEFAULT = SavgolSpec(window=7, polyorder=2)


def quadratic(t: np.ndarray) -> np.ndarray:
    return np.column_stack([0.3 + 1.2 * t - 4.9 * t**2, -2.0 * t, 0.5 + 0.8 * t**2])


def times(n: int = N) -> np.ndarray:
    return np.arange(n) * DT


def test_three_point_window_matches_the_finite_difference_sigma() -> None:
    spec = SavgolSpec(window=3, polyorder=2)
    sigma = np.full(N, 0.02)
    velocity = propagate_linear_sigma(savgol_operator(N, spec, 1, DT), sigma)
    acceleration = propagate_linear_sigma(savgol_operator(N, spec, 2, DT), sigma)
    np.testing.assert_allclose(velocity[1:-1], central_difference_sigma(0.02, DT), rtol=1e-12)
    np.testing.assert_allclose(acceleration, second_difference_sigma(0.02, DT), rtol=1e-12)


def test_quadratic_track_gives_exact_acceleration_at_every_sample() -> None:
    t = times()
    position = quadratic(t)
    out = differentiate(t, position, np.zeros_like(position), np.ones(N, bool), DEFAULT)
    np.testing.assert_allclose(out.acceleration, [[-9.8, 0.0, 1.6]] * N, rtol=0, atol=1e-9)
    velocity = np.column_stack([1.2 - 9.8 * t, np.full(N, -2.0), 1.6 * t])
    np.testing.assert_allclose(out.velocity, velocity, rtol=0, atol=1e-9)
    assert bool((out.acceleration_sigma == 0.0).all())


def test_closed_form_sigma_matches_monte_carlo() -> None:
    rng = np.random.default_rng(7)
    sigma = rng.uniform(0.01, 0.05, N)
    x = quadratic(times())[:, 0]
    draws = {"noise": rng.normal(0.0, sigma, size=(4000, N))}
    for deriv in (1, 2):
        operator = savgol_operator(N, DEFAULT, deriv, DT)

        def model(d: Mapping[str, np.ndarray], w: np.ndarray = operator) -> np.ndarray:
            return np.asarray((w @ (x + d["noise"]).T).T)

        sampled = monte_carlo(model, draws).std
        closed = propagate_linear_sigma(operator, sigma)
        assert float(np.abs(sampled / closed - 1.0).max()) < 0.05, deriv


def test_gaps_split_runs_and_short_runs_are_nan() -> None:
    t = times()
    position = quadratic(t)
    valid = np.ones(N, bool)
    valid[4] = False
    valid[9] = False
    out = differentiate(t, position, np.full_like(position, 0.01), valid, DEFAULT)
    assert bool(np.isnan(out.velocity[[4, 9]]).all())
    alone = differentiate(t[:4], position[:4], np.full((4, 3), 0.01), np.ones(4, bool), DEFAULT)
    np.testing.assert_array_equal(out.velocity[:4], alone.velocity)
    np.testing.assert_array_equal(out.acceleration_sigma[:4], alone.acceleration_sigma)
    assert bool(np.isnan(out.acceleration[10:]).all())
    assert bool(np.isnan(out.velocity_sigma[10:]).all())
    assert bool(np.isfinite(out.acceleration[5:9]).all())


def test_nan_position_sigma_leaves_derivative_sigma_unknown(fixture_clip: SyntheticClip) -> None:
    sampling = sampling_of(fixture_clip.manifest)
    truth = fixture_clip.ground_truth
    lift = align_ground_truth(truth, truth.masks(sampling), sampling).states
    unknown = replace(lift, position_sigma=np.full_like(lift.position, np.nan))
    output = run_provider(
        SavgolKinematics, make_context(fixture_clip, upstream={Stage.LIFT: unknown})
    )
    assert isinstance(output, ObjectStateTable)
    assert bool(np.isnan(output.velocity_sigma).all())
    assert bool(np.isnan(output.acceleration_sigma).all())
    ball = output.object_index("ball")
    np.testing.assert_allclose(output.acceleration[:, ball], [[0.0, 9.81, 0.0]] * N, atol=1e-9)
    np.testing.assert_array_equal(output.position, lift.position)
