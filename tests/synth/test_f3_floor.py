"""f3_floor: seeded draws, balls rolling on the floor, one horizontal impact, no bounce after it."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from physverify.core.errors import ConfigError
from physverify.schema.scene import SceneSpec
from physverify.synth.ballistics import GravitySchedule, energy_scale, integrate_sphere
from physverify.synth.family_toolkit import clip_times
from physverify.synth.seeds import scene_rng
from physverify.synth.settings import DatasetSettings, family_params

NAME = "f3_floor"
INDICES = range(8)


@pytest.fixture
def params(ds: DatasetSettings, families: Any, priors: dict[str, Any]) -> Any:
    return family_params(families[NAME], ds, dict(priors))


def _sample(families: Any, params: Any, seed: int, index: int) -> SceneSpec:
    scene: SceneSpec = families[NAME]().sample(scene_rng(seed, NAME, index), params)
    return scene


def test_same_seed_same_scene_and_seeds_differ(families: Any, params: Any) -> None:
    first = [_sample(families, params, 7, i) for i in INDICES]
    again = [_sample(families, params, 7, i) for i in INDICES]
    other = _sample(families, params, 8, 0)
    assert first == again
    assert len({s.objects[0].position for s in first}) == len(first)
    assert other.objects[0].position != first[0].objects[0].position


def test_balls_are_equal_and_roll_on_the_floor(families: Any, params: Any) -> None:
    for i in INDICES:
        a, b = _sample(families, params, 7, i).objects
        assert a.extents == b.extents and a.restitution == b.restitution
        low, high = params.restitution_range
        assert low <= a.restitution <= high
        radius = a.extents[0] / 2.0
        for ball in (a, b):
            assert ball.position[2] == pytest.approx(radius, abs=1e-12)
            assert ball.velocity[2] == 0.0
            contact = np.asarray(ball.velocity) + np.cross(
                ball.angular_velocity, [0.0, 0.0, -radius]
            )
            assert np.allclose(contact, 0.0, atol=1e-12)


def test_too_small_start_gap_is_rejected(families: Any, params: Any) -> None:
    tight = params.model_copy(
        update={"relative_speed_range": (0.01, 0.01), "collision_time_range": (0.3, 0.3)}
    )
    with pytest.raises(ConfigError, match="start too close"):
        _sample(families, tight, 7, 0)


@pytest.mark.parametrize("t_c", [0.30, 0.37, 0.45])
def test_balls_first_touch_at_the_drawn_time_along_a_horizontal_normal(
    families: Any, params: Any, collide: Any, t_c: float
) -> None:
    fixed = params.model_copy(update={"collision_time_range": (t_c, t_c)})
    for i in INDICES:
        a, b = _sample(families, fixed, 7, i).objects
        hit = collide(a, b, np.zeros(3), 0.0)
        assert hit is not None
        assert hit.t == pytest.approx(t_c, abs=1e-9)
        assert hit.normal[2] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("gain", [0.0, 1.0])
def test_no_floor_bounce_after_the_collision(
    families: Any, params: Any, collide: Any, gain: float
) -> None:
    """The clean impulse-law states (gain 0) and the r3_energy_gain rescaling at dE/E = 1 fly on
    as rolling: no floor impact and the centre stays at the rest height."""
    g = np.array([0.0, 0.0, -9.81])
    for i in INDICES:
        scene = _sample(families, params, 7, i)
        a, b = scene.objects
        hit = collide(a, b, np.zeros(3), 0.0)
        before = [np.asarray(a.velocity), np.asarray(b.velocity)]
        after = [hit.after_a[1], hit.after_b[1]]
        masses = (a.mass, b.mass)
        m = np.asarray(masses)
        cm = m @ np.array(before) / m.sum()
        scale = 1.0 if gain == 0.0 else energy_scale(masses, before, after, gain)
        times = clip_times(params)
        for (position, _), v in zip((hit.after_a, hit.after_b), after, strict=True):
            velocity = cm + scale * (v - cm)
            assert velocity[2] == pytest.approx(0.0, abs=1e-12)
            radius = a.extents[0] / 2.0
            flight = integrate_sphere(
                position,
                velocity,
                hit.t,
                radius,
                GravitySchedule.constant(g),
                a.restitution,
                times[times >= hit.t],
            )
            assert flight.bounces == []
            assert np.allclose(flight.positions[:, 2], radius, atol=1e-9)
