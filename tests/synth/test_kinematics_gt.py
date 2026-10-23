"""WP-G 1.C: the position-consistent GT velocity on the recorded Bullet clips."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from physverify.schema.constants import ENV_ID
from physverify.synth.gt_velocity import (
    corrected_cells,
    euler_velocity_offset,
    position_consistent,
    resting_cells,
)

FAMILIES = ("f1_projectile", "f2_bounce", "f3_collision", "f4_stack", "f5_contact")


def flight_steps(clip: Any) -> list[tuple[int, int]]:
    """(frame, object) pairs whose step to the next frame is free flight: both cells corrected and
    no contact record of the object within a frame of either end."""
    states, contacts = clip.states, clip.contacts
    ids = states.object_ids.astype(str).tolist()
    cells = corrected_cells(states, contacts, clip.scene)
    near = np.zeros_like(cells)
    for record in contacts.records:
        lo = max(0, int(np.floor(record.t_start)) - 2)
        hi = int(np.ceil(record.t_end)) + 2
        for object_id in (record.object_a, record.object_b):
            if object_id != ENV_ID:
                near[lo:hi, ids.index(object_id)] = True
    ok = cells[:-1] & cells[1:] & ~near[:-1] & ~near[1:]
    return [(int(k), int(i)) for k, i in zip(*np.nonzero(ok), strict=True)]


def parabola_residual(clip: Any, velocity: np.ndarray, steps: list[tuple[int, int]]) -> float:
    """max |x_{k+1} - x_k - v_k dt - g dt^2 / 2| over the steps, with the scene's gravity."""
    states = clip.states
    g = np.asarray(clip.scene.gravity, dtype=np.float64)
    worst = 0.0
    for k, i in steps:
        dt = float(states.t[k + 1] - states.t[k])
        jump = states.position[k + 1, i] - states.position[k, i]
        residual = jump - velocity[k, i] * dt - 0.5 * g * dt**2
        worst = max(worst, float(np.abs(residual).max()))
    return worst


def check_identity(clip: Any) -> None:
    steps = flight_steps(clip)
    assert steps, "the clip has no free-flight step to test"
    fixed = position_consistent(clip.states, clip.contacts, clip.scene)
    dt = 1.0 / clip.scene.frame_rate
    offset = float(np.abs(euler_velocity_offset(clip.scene)).max())
    assert parabola_residual(clip, fixed.velocity, steps) < 1e-9
    assert parabola_residual(clip, clip.states.velocity, steps) == pytest.approx(
        offset * dt, rel=1e-6
    )


@pytest.mark.parametrize("family", ["f1_projectile", "f2_bounce", "f3_collision"])
def test_the_corrected_parabola_passes_through_the_euler_positions(bank: Any, family: str) -> None:
    check_identity(bank.recorded(family))


@pytest.mark.parametrize("family", ["f1_projectile", "f2_bounce"])
def test_a_gravity_scaled_clip_is_corrected_by_k_g(bank: Any, family: str) -> None:
    clip, record = bank.recorded_gravity_scale(family)
    g = np.asarray(clip.scene.gravity, dtype=np.float64)
    k = float(np.linalg.norm(g)) / 9.81
    assert abs(k - 1.0) == pytest.approx(record.magnitude, rel=1e-6)
    assert euler_velocity_offset(clip.scene) == pytest.approx(g / (2 * clip.scene.step_rate))
    check_identity(clip)


@pytest.mark.parametrize("family", FAMILIES)
def test_only_flying_cells_move_and_nothing_else_changes(bank: Any, family: str) -> None:
    clip = bank.recorded(family)
    states = clip.states
    fixed = position_consistent(states, clip.contacts, clip.scene)
    cells = corrected_cells(states, clip.contacts, clip.scene)
    ids = states.object_ids.astype(str).tolist()
    resting = resting_cells(clip.contacts, ids, states.n_frames)
    static = [ids.index(o.object_id) for o in clip.scene.objects if o.static]
    assert not (cells & resting).any()
    assert not cells[:, static].any()
    np.testing.assert_array_equal(fixed.velocity[~cells], states.velocity[~cells])
    shift = fixed.velocity[cells] - states.velocity[cells]
    np.testing.assert_allclose(
        shift, np.broadcast_to(euler_velocity_offset(clip.scene), shift.shape), atol=1e-12
    )
    for field in dataclasses.fields(states):
        if field.name == "velocity":
            continue
        before, after = getattr(states, field.name), getattr(fixed, field.name)
        if isinstance(before, np.ndarray):
            assert np.array_equal(before, after, equal_nan=before.dtype.kind == "f"), field.name
        else:
            assert before == after, field.name


@pytest.mark.parametrize("family", ["f4_stack", "f5_contact"])
def test_resting_objects_keep_the_simulator_velocity(bank: Any, family: str) -> None:
    clip = bank.recorded(family)
    ids = clip.states.object_ids.astype(str).tolist()
    resting = resting_cells(clip.contacts, ids, clip.states.n_frames)
    assert resting.any()
    fixed = position_consistent(clip.states, clip.contacts, clip.scene)
    np.testing.assert_array_equal(fixed.velocity[resting], clip.states.velocity[resting])
