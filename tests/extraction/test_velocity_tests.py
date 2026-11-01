"""Velocity tests of ``contacts/kinematic_scene``: sigma_v from the profile, rest runs and the
significance of a velocity change along a normal."""

from __future__ import annotations

import math
from dataclasses import replace

import contact_scenes as cs
import numpy as np
import pytest
import scene_contact_cases as cases
from scipy.stats import binom, chi2, norm

from physverify.extraction.providers._impacts import Bases, ImpactRows, track_impacts
from physverify.extraction.providers._proximity import object_lengths
from physverify.extraction.providers._velocity_tests import (
    NormalChange,
    VelocityError,
    long_runs,
    rest_rows,
    rest_runs,
)
from physverify.physics.kinematics import reference_speed
from physverify.physics.row_time import reference_seconds
from physverify.physics.tracks import NoiseScale
from physverify.schema.constants import ENV_ID, STANDARD_GRAVITY
from physverify.schema.contacts import ContactKind
from physverify.schema.stages import Stage

sc = cs.sc
ALPHA = 0.05
SIGMA = 0.2
MIN_ROWS = 4
UP = np.array([0.0, 0.0, 1.0])


def visible(rows: int) -> np.ndarray:
    return np.ones(rows, dtype=np.bool_)


# 1. sigma_v


def test_sigma_v_is_the_entry_rms_times_the_reference_speed() -> None:
    velocity = np.random.default_rng(0).normal(0.0, 1.0, (30, 3))
    error = VelocityError.of(cases.VEL_REL)
    expected = math.hypot(-0.0022, 0.3157) * reference_speed(velocity, 0.2, 1.3)
    assert error.sigma(velocity, 0.2, 1.3) == pytest.approx(expected, rel=1e-12)
    assert error.known


def test_an_unknown_entry_is_nan_and_the_provider_returns_an_empty_table() -> None:
    error = VelocityError.of(None)
    assert not error.known and math.isnan(error.sigma(np.ones((5, 3)), 0.2, 1.0))
    case = cs.bounce_on_floor()
    entries = cases.measured_entries()
    assert cases.scene_table(case, entries).records
    del entries[(Stage.KINEMATICS, "vel_rel")]
    assert cases.scene_table(case, entries).records == ()


# 2. Rest


def test_a_still_object_with_small_noise_is_one_run() -> None:
    velocity = np.random.default_rng(1).normal(0.0, 0.3 * SIGMA, (40, 3))
    assert rest_runs(velocity, visible(40), SIGMA, ALPHA, MIN_ROWS) == (slice(0, 40),)


def test_a_speed_of_three_sigma_is_never_at_rest() -> None:
    velocity = np.tile([3.0 * SIGMA, 0.0, 0.0], (40, 1))
    assert not rest_rows(velocity, visible(40), SIGMA, ALPHA).any()


def test_the_apex_of_a_vertical_throw_passes_row_by_row_but_is_not_a_run() -> None:
    t = np.linspace(0.0, 1.0, 41)
    velocity = np.zeros((41, 3))
    velocity[:, 2] = 5.0 - 10.0 * t
    sigma = 1.0
    rows = rest_rows(velocity, visible(41), sigma, ALPHA)
    assert rows.sum() >= MIN_ROWS
    assert rest_runs(velocity, visible(41), sigma, ALPHA, MIN_ROWS) == ()


def test_a_run_shorter_than_the_minimum_is_nothing() -> None:
    velocity = np.tile([1.0, 0.0, 0.0], (20, 1))
    velocity[8:11] = 0.0
    assert rest_rows(velocity, visible(20), SIGMA, ALPHA).sum() == 3
    assert rest_runs(velocity, visible(20), SIGMA, ALPHA, MIN_ROWS) == ()


def test_an_unknown_sigma_has_no_rest() -> None:
    assert not rest_rows(np.zeros((10, 3)), visible(10), math.nan, ALPHA).any()


def scalar_rest_runs(
    velocity: np.ndarray, rows_visible: np.ndarray, sigma: float, min_rows: int
) -> tuple[slice, ...]:
    """The 3D rest runs before sigma could vary per row (chi2_3, 2 sigma^2 constancy)."""
    quantile = float(chi2.isf(ALPHA, 3))
    with np.errstate(invalid="ignore"):
        rows = (np.sum(velocity**2, axis=-1) <= quantile * sigma**2) & rows_visible
    kept = []
    for run in long_runs(rows, min_rows):
        change = velocity[run.stop - 1] - velocity[run.start]
        if float(change @ change) <= quantile * 2.0 * sigma**2:
            kept.append(run)
    return tuple(kept)


@pytest.mark.parametrize("seed", range(40))
def test_scalar_sigma_runs_in_space_are_the_old_runs(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = int(rng.integers(5, 80))
    sigma = float(rng.uniform(0.05, 1.0))
    velocity = rng.normal(0.0, rng.uniform(0.2, 2.0) * sigma, (rows, 3))
    velocity[rng.random(rows) < 0.2] *= 6.0
    shown = rng.random(rows) > 0.1
    expected = scalar_rest_runs(velocity, shown, sigma, MIN_ROWS)
    assert rest_runs(velocity, shown, sigma, ALPHA, MIN_ROWS) == expected
    per_row = np.full(rows, sigma)
    assert rest_runs(velocity, shown, per_row, ALPHA, MIN_ROWS) == expected


def test_image_velocities_use_two_degrees_of_freedom() -> None:
    edge = math.sqrt(chi2.isf(ALPHA, 2)) * SIGMA
    velocity = np.array([[edge * 0.999, 0.0], [edge * 1.001, 0.0]])
    assert rest_rows(velocity, visible(2), SIGMA, ALPHA).tolist() == [True, False]


def test_a_per_row_sigma_scales_each_row_and_the_run_ends() -> None:
    velocity = np.tile([1.0, 0.0], (6, 1))
    sigma = np.array([0.1, 1.0, 1.0, 1.0, 1.0, 0.1])
    assert rest_rows(velocity, visible(6), sigma, ALPHA).tolist() == [False, *[True] * 4, False]
    swing = np.zeros((6, 2))
    swing[0, 0], swing[-1, 0] = -0.5, 0.5
    tight = np.full(6, 0.25)
    loose = np.array([1.0, 0.25, 0.25, 0.25, 0.25, 1.0])
    assert rest_rows(swing, visible(6), tight, ALPHA).all()
    assert rest_runs(swing, visible(6), tight, ALPHA, MIN_ROWS) == ()
    assert rest_runs(swing, visible(6), loose, ALPHA, MIN_ROWS) == (slice(0, 6),)


# 3. Significance along the normal


def test_the_bound_is_the_one_sided_bonferroni_quantile() -> None:
    one = NormalChange.along(np.zeros(3), UP, SIGMA, 1)
    assert one.bound(ALPHA) == pytest.approx(1.6448536269514722 * math.sqrt(2.0) * SIGMA)
    assert one.bound(ALPHA) == pytest.approx(norm.isf(0.05) * math.sqrt(2.0) * SIGMA, rel=1e-15)
    four = NormalChange.along(np.zeros(3), UP, SIGMA, 4)
    assert four.bound(ALPHA) == pytest.approx(norm.isf(0.0125) * math.sqrt(2.0) * SIGMA, rel=1e-15)


@pytest.mark.parametrize(("ratio", "accepted"), [(5.0, True), (1.0, False), (-5.0, False)])
def test_a_change_passes_only_upwards_beyond_the_bound(ratio: float, accepted: bool) -> None:
    sigma_du = math.sqrt(2.0) * SIGMA
    change = NormalChange.along(np.array([0.3, -0.1, ratio * sigma_du]), UP, SIGMA, 1)
    assert change.ratio == pytest.approx(ratio)
    assert change.significant(ALPHA) is accepted
    assert change.confidence == pytest.approx(norm.cdf(ratio))


def ball_ratio_scale(case: sc.Case) -> float:
    """e_v giving the ball's environment change du / sigma_du = 1 (du = (1 + e) v_impact)."""
    unit_sigma = VelocityError(1.0).sigmas(case.states, object_lengths(case.states))
    return (1.0 + 0.8) * 5.0 / (math.sqrt(2.0) * float(unit_sigma[0]))


@pytest.mark.parametrize(("ratio", "found"), [(5.0, True), (1.0, False)])
def test_a_bounce_is_an_impact_only_when_significant(ratio: float, found: bool) -> None:
    case = cs.bounce_on_floor()
    outcome = cases.outcome(case, velocity=VelocityError(ball_ratio_scale(case) / ratio))
    impacts = [
        r
        for r in outcome.records
        if (r.object_a, r.object_b, r.kind) == ("ball", ENV_ID, ContactKind.IMPACT)
    ]
    assert len(impacts) == int(found)
    tested = [t for t in outcome.tested if t.change is not None and t.change.delta > 1.0]
    assert [t.change.ratio for t in tested if t.change] == pytest.approx([ratio], rel=0.05)


def test_a_teleport_has_no_impact() -> None:
    case = sc.teleport(pitch_deg=25.0)
    floor = sc.build("box_only", [cs.floor_box(x=1.5)], [], 25.0).states
    case = replace(case, states=cs.joined(case.states, floor))
    records = cases.outcome(case).records
    assert not [r for r in records if r.kind is ContactKind.IMPACT]


# 4. Spurious impacts under H0


def parabolas(seed: int, count: int, sigma_x: float) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    t = sc.times()
    tracks = []
    for _ in range(count):
        v0 = rng.uniform(-2.0, 2.0, 3)
        clean = v0 * t[:, None] - 0.5 * STANDARD_GRAVITY * UP * t[:, None] ** 2
        tracks.append(clean + rng.normal(0.0, sigma_x, clean.shape))
    return tracks


def candidate_changes(track: np.ndarray, sigma_x: float, bases: Bases) -> list[float]:
    rows = ImpactRows.at(reference_seconds(4), sc.DT)
    noise = NoiseScale(np.full(3, sigma_x), known=True)
    found = track_impacts(sc.times(), track, noise, rows, ALPHA, bases, bonferroni=False)
    return [float(i.delta_v @ UP) for i in found]


def test_clean_tracks_have_at_most_alpha_false_impacts_per_track() -> None:
    """sigma_v is set from the candidates of other clean tracks (the noise the gate faces);
    300 fresh tracks then reject H0 at most at the binomial 95 % bound of alpha."""
    sigma_x, count = 0.002, 300
    bases: Bases = {}
    calibration = [
        du
        for track in parabolas(7, count // 3, sigma_x)
        for du in candidate_changes(track, sigma_x, bases)
    ]
    sigma_v = float(np.sqrt(np.mean(np.square(calibration)))) / math.sqrt(2.0)
    false = 0
    for track in parabolas(8, count, sigma_x):
        changes = candidate_changes(track, sigma_x, bases)
        tests = [NormalChange.along(du * UP, UP, sigma_v, len(changes)) for du in changes]
        false += any(change.significant(ALPHA) for change in tests)
    assert false <= binom.ppf(0.95, count, ALPHA)


def test_a_noise_only_object_alone_has_no_environment_record() -> None:
    case = sc.build("alone", [cases.box("box0", 0.0, 0.0)], [], 25.0)
    states = cs.jittered(case.states, 0.002, seed=5)
    noisy = replace(
        states,
        velocity=states.velocity
        + np.random.default_rng(6).normal(0.0, 0.01, states.velocity.shape),
    )
    assert cases.env_records(cases.outcome(case, states=noisy).records) == []
