"""R4_jump rigid2_dfloor: the depth axis of sigma-hat floored at the declared depth sigma."""

from __future__ import annotations

import math
from dataclasses import replace

import check_scenarios as sc
import numpy as np
import pytest

from physverify.checks._base import CheckInputs
from physverify.checks.r4_track_jump import NoiseContinuityCheck, NoiseJumpParams
from physverify.checks.variants._declared_depth_jump import (
    DEPTH_AXIS,
    DeclaredDepthJumpCheck,
    declared_depth_sigma,
)
from physverify.checks.variants._rigid_jump import RigidJumpCheck, RigidJumpParams
from physverify.schema.checks import CheckResult
from physverify.schema.states import CoordFrame

SIGMA = 0.01
DEPTH_SIGMA = 0.39 * 4.0
"""About the depth sigma a monocular lift declares at the scenes' depth of 4 m."""
ROW = 11


def rigid(inputs: CheckInputs) -> CheckResult:
    (result,) = RigidJumpCheck(RigidJumpParams()).evaluate(inputs)
    return result


def dfloor(inputs: CheckInputs) -> CheckResult:
    (result,) = DeclaredDepthJumpCheck(RigidJumpParams()).evaluate(inputs)
    return result


def noisy_rest(declared: np.ndarray, seed: int = 5, frames: int = 24) -> CheckInputs:
    """A ball at rest with white position noise SIGMA and the per-axis sigma ``declared``."""
    inputs = sc.rolling_ball(speed=0.0, frames=frames).inputs()
    states = inputs.states
    noise = np.random.default_rng(seed).normal(0.0, SIGMA, states.position.shape)
    sigma = np.broadcast_to(declared, states.position_sigma.shape).copy()
    states = replace(states, position=states.position + noise, position_sigma=sigma)
    return replace(inputs, states=states)


def stepped(inputs: CheckInputs, axis: int, size: float) -> CheckInputs:
    position = inputs.states.position.copy()
    position[ROW + 1 :, 0, axis] += size
    return replace(inputs, states=replace(inputs.states, position=position))


DECLARED = np.array([SIGMA, SIGMA, DEPTH_SIGMA])


def test_a_jump_in_depth_under_the_declared_sigma_is_not_seen() -> None:
    inputs = stepped(noisy_rest(DECLARED), DEPTH_AXIS, 50 * SIGMA)
    seen, floored = rigid(inputs), dfloor(inputs)
    assert seen.magnitude > 10.0
    assert floored.magnitude < seen.magnitude / 3
    assert floored.details["noise_source"] == "track"


def test_a_jump_across_the_line_of_sight_is_seen_as_by_rigid2() -> None:
    inputs = stepped(noisy_rest(DECLARED), 0, 50 * SIGMA)
    seen, floored = rigid(inputs), dfloor(inputs)
    assert seen.magnitude > 10.0
    assert 0.9 * seen.magnitude <= floored.magnitude <= seen.magnitude
    assert floored.details["row"] == seen.details["row"] == ROW


def test_exact_positions_keep_the_exact_noise_of_rigid2() -> None:
    inputs = sc.teleport().inputs()
    seen, floored = rigid(inputs), dfloor(inputs)
    assert floored.details["noise_source"] == seen.details["noise_source"] == "exact"
    assert floored.magnitude == seen.magnitude and floored.sigma == seen.sigma == 0.0


def test_an_unknown_declared_sigma_keeps_the_pixel_floor() -> None:
    inputs = stepped(noisy_rest(np.full(3, math.nan)), DEPTH_AXIS, 50 * SIGMA)
    seen, floored = rigid(inputs), dfloor(inputs)
    assert floored.magnitude == seen.magnitude
    assert floored.details == seen.details


def test_the_default_floor_is_one_pixel_on_every_axis() -> None:
    inputs = noisy_rest(DECLARED)
    excluded = np.zeros(inputs.states.n_frames, dtype=bool)
    floor = NoiseContinuityCheck(NoiseJumpParams())._floor(inputs, 0, excluded, 0.003)
    assert floor.tolist() == [0.003, 0.003, 0.003]


def test_the_floor_reads_the_declared_depth_sigma_on_the_kept_rows() -> None:
    sigma = np.tile(DECLARED, (24, 1))
    sigma[:6, DEPTH_AXIS] = 100.0
    inputs = noisy_rest(sigma[:, None, :])
    excluded = np.zeros(24, dtype=bool)
    excluded[:6] = True
    check = DeclaredDepthJumpCheck(RigidJumpParams())
    assert declared_depth_sigma(inputs.states, 0, excluded) == pytest.approx(DEPTH_SIGMA)
    floor = check._floor(inputs, 0, excluded, 0.003)
    assert floor.tolist() == pytest.approx([0.003, 0.003, DEPTH_SIGMA])


def test_another_frame_declares_no_depth_sigma() -> None:
    inputs = noisy_rest(DECLARED)
    world = replace(inputs.states, frame=CoordFrame.WORLD_ZUP)
    assert math.isnan(declared_depth_sigma(world, 0, np.zeros(24, dtype=bool)))
