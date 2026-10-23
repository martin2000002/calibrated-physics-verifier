"""A floor bounce whose row after the crossing is already under the floor is solved from
the row before it; every other bounce resolves exactly as before."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from reference.legacy_contact_rules import bounce_case

from physverify.physics.measures.collision_energy import restitution
from physverify.schema.contacts import ContactTable
from physverify.schema.states import ObjectStateTable
from physverify.synth.ballistics import impact_velocity, time_to_height
from physverify.synth.frame_windows import UP
from physverify.synth.magnitudes import (
    bounce_crossing,
    bounce_restitution,
    floor_bounce,
    floor_crossing,
)
from physverify.synth.perturbations._state_edit import (
    edit_object,
    first_floor_impact,
    gravity_of,
    radius_of,
    restitution_for_edit,
    spheres,
)

TOL = 1e-12
Vec = np.ndarray


def legacy_floor_bounce(
    states: ObjectStateTable, index: int, t_guess: float, g: Vec
) -> tuple[float, Vec, Vec] | None:
    """``floor_bounce`` before the fix: a row k already under the floor fell back to t_c = t_k."""
    vz = states.velocity[:, index, 2]
    radius = radius_of(states, index)
    base = math.floor(t_guess)
    for k in (base, base - 1, base + 1):
        if 0 <= k < states.n_frames - 1 and vz[k] < 0.0 < vz[k + 1]:
            t_k, t_next = float(states.t[k]), float(states.t[k + 1])
            p, v = states.position[k, index], states.velocity[k, index]
            tau = time_to_height(float(p[2]), float(v[2]), float(g[2]), radius)
            t_c = t_k + tau if math.isfinite(tau) else t_k
            if not t_k <= t_c <= t_next:
                continue
            v_minus = impact_velocity(v, t_k, t_c, g)
            v_plus = impact_velocity(states.velocity[k + 1, index], t_next, t_c, g)
            return t_c, v_minus, v_plus
    return None


def ball(states: ObjectStateTable, contacts: ContactTable) -> tuple[int, float]:
    """The bouncing sphere and the native frame of its first floor impact."""
    index = next(i for i in spheres(states) if first_floor_impact(_id(states, i), contacts))
    impact = first_floor_impact(_id(states, index), contacts)
    assert impact is not None
    return index, impact.t_start


def penetrated_twin(clip: Any) -> tuple[ObjectStateTable, int, int, float]:
    """The analytic f2 clip with the first row after the bounce replaced by the incoming arc
    continued from the row before: that row is under the floor and still falling (the Bullet
    pattern the old fallback mishandled)."""
    states, g = clip.states, gravity_of(clip.scene)
    index, frame = ball(states, clip.contacts)
    t_b = frame / float(clip.scene.frame_rate)
    k = int(np.searchsorted(states.t, t_b, side="right"))
    dt = float(states.t[k] - states.t[k - 1])
    p, v = states.position[k - 1, index], states.velocity[k - 1, index]
    twin = edit_object(
        states,
        index,
        slice(k, k + 1),
        position=(p + v * dt + 0.5 * g * dt**2)[None],
        velocity=(v + g * dt)[None],
    )
    assert twin.position[k, index, 2] < radius_of(twin, index)
    assert twin.velocity[k, index, 2] < 0.0 < twin.velocity[k + 1, index, 2]
    return twin, index, k, t_b


@pytest.fixture(scope="module")
def f2(bank: Any) -> Any:
    return bank.analytic("f2_bounce")


def test_the_twin_shows_the_old_fallback(f2: Any) -> None:
    twin, index, k, t_b = penetrated_twin(f2)
    old = legacy_floor_bounce(twin, index, float(k), gravity_of(f2.scene))
    assert old is not None and old[0] == float(twin.t[k]) != pytest.approx(t_b, abs=1e-6)


@pytest.mark.parametrize("guess", ["k", "k-1", "t_b"])
def test_a_penetrated_row_is_solved_from_the_row_before(f2: Any, guess: str) -> None:
    twin, index, k, t_b = penetrated_twin(f2)
    g = gravity_of(f2.scene)
    t_guess = {"k": float(k), "k-1": float(k - 1), "t_b": t_b * float(f2.scene.frame_rate)}[guess]
    found = floor_bounce(twin, index, t_guess, g)
    assert found is not None
    t_c, v_minus, v_plus = found
    assert bounce_crossing(twin, index, t_guess, g) == (k, k - 1, t_c)
    assert t_c == pytest.approx(t_b, abs=TOL)
    exact_minus = impact_velocity(twin.velocity[k - 1, index], float(twin.t[k - 1]), t_b, g)
    exact_plus = impact_velocity(twin.velocity[k + 1, index], float(twin.t[k + 1]), t_b, g)
    np.testing.assert_allclose(v_minus, exact_minus, rtol=0.0, atol=TOL)
    np.testing.assert_allclose(v_plus, exact_plus, rtol=0.0, atol=TOL)
    e = next(o.restitution for o in f2.scene.objects if o.object_id == _id(twin, index))
    assert restitution(v_minus, v_plus, UP) == pytest.approx(e, abs=TOL)


@pytest.mark.parametrize(
    ("source", "family"),
    [
        ("analytic", "f2_bounce"),
        ("recorded", "f2_bounce"),
        ("recorded", "f3_collision"),
        ("recorded", "f1_projectile"),
    ],
)
def test_a_bounce_above_the_floor_resolves_bit_for_bit_as_before(
    bank: Any, source: str, family: str
) -> None:
    clip = getattr(bank, source)(family)
    g = gravity_of(clip.scene)
    compared = 0
    for index in spheres(clip.states):
        impact = first_floor_impact(_id(clip.states, index), clip.contacts)
        if impact is None or bounce_case(clip.states, index, impact.t_start, g).penetrated:
            continue
        new = floor_bounce(clip.states, index, impact.t_start, g)
        old = legacy_floor_bounce(clip.states, index, impact.t_start, g)
        assert new is not None and old is not None
        assert new[0] == old[0]
        assert np.array_equal(new[1], old[1]) and np.array_equal(new[2], old[2])
        compared += 1
    assert compared > 0


def test_a_row_exactly_on_the_floor_counts_as_penetrated(f2: Any) -> None:
    twin, index, k, _ = penetrated_twin(f2)
    radius = radius_of(twin, index)
    on_floor = edit_object(
        twin, index, slice(k, k + 1), position=np.r_[twin.position[k, index, :2], radius][None]
    )
    crossing = floor_crossing(on_floor, index, k, radius, float(gravity_of(f2.scene)[2]))
    assert crossing is not None and crossing[0] == k - 1
    assert float(twin.t[k - 1]) < crossing[1] <= float(twin.t[k])


def test_a_penetrated_first_row_has_no_row_before(f2: Any) -> None:
    states = f2.states
    index, _ = ball(states, f2.contacts)
    radius = radius_of(states, index)
    sunk = edit_object(
        states, index, slice(0, 1), position=np.r_[states.position[0, index, :2], radius / 2][None]
    )
    assert floor_crossing(sunk, index, 0, radius, float(gravity_of(f2.scene)[2])) is None


def test_the_bounce_case_names_the_row_the_crossing_comes_from(f2: Any) -> None:
    g = gravity_of(f2.scene)
    twin, index, k, t_b = penetrated_twin(f2)
    moved = bounce_case(twin, index, float(k), g)
    assert (moved.k, moved.penetrated, moved.status) == (k, True, "previous_row")
    assert moved.depth_m < 0.0
    clean = bounce_case(f2.states, index, float(k), g)
    assert (clean.k, clean.penetrated, clean.status) == (k - 1, False, "same_row")


def test_the_restitution_edit_flies_from_the_crossing_on_a_penetrated_twin(
    f2: Any, plugins: dict[str, Any]
) -> None:
    twin, index, k, t_b = penetrated_twin(f2)
    plugin = plugins["r3_restitution_gain"]
    magnitude = plugin.params.levels[0]
    moved, contacts, record = plugin.apply(
        f2.scene, twin, f2.contacts, magnitude, np.random.default_rng(0)
    )
    fps = float(f2.scene.frame_rate)
    assert record.params["impact_frame"] == pytest.approx(t_b * fps, abs=1e-9)
    e = next(o.restitution for o in f2.scene.objects if o.object_id == _id(twin, index))
    assert record.params["e_clean"] == pytest.approx(e, abs=1e-9)
    assert np.all(moved.position[k:, index, 2] >= radius_of(moved, index) - TOL)
    assert plugin.measure(moved, contacts, record) == pytest.approx(magnitude, abs=1e-9)


def test_the_recorded_f1_ball_takes_its_restitution_from_the_crossing_before_row_17(
    bank: Any,
) -> None:
    clip = bank.recorded("f1_projectile")
    states, g = clip.states, gravity_of(clip.scene)
    index = states.object_index("ball_0")
    impact = first_floor_impact("ball_0", clip.contacts)
    assert impact is not None
    found = bounce_crossing(states, index, impact.t_start, g)
    assert found is not None and found[:2] == (17, 16)
    radius = radius_of(states, index)
    assert states.position[17, index, 2] < radius < states.position[16, index, 2]
    z, vz, gz = states.position[16, index, 2], states.velocity[16, index, 2], g[2]
    roots = np.roots([0.5 * gz, vz, z - radius])
    tau = float(min(r.real for r in roots if abs(r.imag) == 0.0 and r.real > 0.0))
    t_c = float(states.t[16]) + tau
    v_minus = vz + gz * tau
    v_plus = states.velocity[18, index, 2] + gz * (t_c - float(states.t[18]))
    by_hand = -v_plus / v_minus
    assert bounce_restitution(states, index, impact.t_start, g) == pytest.approx(by_hand, abs=TOL)
    edit = restitution_for_edit(clip.scene, states, clip.contacts, index)
    assert edit == pytest.approx(by_hand, abs=TOL)
    old = legacy_floor_bounce(states, index, impact.t_start, g)
    assert old is not None and restitution(old[1], old[2], UP) != pytest.approx(by_hand, abs=1e-6)


def _id(states: ObjectStateTable, index: int) -> str:
    return str(states.object_ids[index])
