"""ADR 0053: f6 scenes keep a regulation ball at >= 24 px over a >= 0.7 s flight, and only the
known-size R5b perturbation applies to them."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from physverify.core.errors import ConfigError
from physverify.schema.scene import SceneSpec
from physverify.schema.states import PrimitiveKind
from physverify.synth.families import f6_known_size as f6
from physverify.synth.family_toolkit import gravity_vector
from physverify.synth.generate import base_scene, split_of
from physverify.synth.plan import ScenePlan, scene_id_of
from physverify.synth.settings import DatasetSettings, family_params

INDICES = range(8)
PERTURBATION = "r5b_known_size_gravity_scale"


@pytest.fixture
def params(ds: DatasetSettings, families: Any, priors: dict[str, Any]) -> f6.Params:
    chosen = family_params(families[f6.NAME], ds, priors)
    assert isinstance(chosen, f6.Params)
    return chosen


@pytest.fixture
def scenes(ds: DatasetSettings, families: Any, priors: dict[str, Any]) -> list[SceneSpec]:
    plans = [
        ScenePlan(scene_id_of(f6.NAME, i), f6.NAME, i, split_of(ds, f6.NAME, i)) for i in INDICES
    ]
    return [base_scene(plan, families[f6.NAME], ds, priors) for plan in plans]


def test_the_family_films_its_own_clip_whatever_the_profile(
    params: f6.Params, scenes: list[SceneSpec]
) -> None:
    for scene in scenes:
        assert scene.n_frames == params.clip_frames
        assert scene.resolution == (params.clip_resolution, params.clip_resolution)


def test_each_scene_is_one_regulation_basketball(
    params: f6.Params, scenes: list[SceneSpec]
) -> None:
    low, high = params.diameter_range_m
    for scene in scenes:
        (ball,) = scene.objects
        assert ball.primitive is PrimitiveKind.SPHERE
        assert ball.label == "basketball"
        assert low <= ball.extents[0] <= high


def test_the_clean_flight_lasts_at_least_the_minimum(
    params: f6.Params, scenes: list[SceneSpec]
) -> None:
    g = -float(gravity_vector()[2])
    for scene in scenes:
        (ball,) = scene.objects
        height = ball.position[2] - ball.extents[0] / 2.0
        vz = ball.velocity[2]
        flight = (vz + math.sqrt(vz**2 + 2.0 * g * height)) / g
        assert flight >= params.min_flight_s - 1e-9


def test_the_ball_is_never_smaller_than_the_minimum_apparent_diameter(
    params: f6.Params, scenes: list[SceneSpec]
) -> None:
    chosen = f6.own_clip(params)
    for scene in scenes:
        (ball,) = scene.objects
        paths = f6.envelope_paths(ball, chosen)
        assert len(paths[0]) == params.clip_frames
        smallest = f6.min_apparent_diameter(scene, np.vstack(paths), ball.extents[0])
        assert smallest >= params.min_diameter_px


def test_launch_speed_closes_the_flight_at_the_floor() -> None:
    g = -float(gravity_vector()[2])
    flight, height = 0.8, 0.2
    vz = f6.launch_speed(flight, height)
    assert height + vz * flight - 0.5 * g * flight**2 == pytest.approx(0.0, abs=1e-12)


def test_a_flight_range_below_the_minimum_is_a_config_error(
    families: Any, params: f6.Params
) -> None:
    short = params.model_copy(update={"flight_s_range": (0.5, 0.9)})
    with pytest.raises(ConfigError):
        families[f6.NAME]().sample(np.random.default_rng(0), short)


def test_only_the_known_size_gravity_scale_applies_to_f6(plugins: dict[str, Any]) -> None:
    applicable = {name for name, p in plugins.items() if f6.NAME in p.families}
    assert applicable == {PERTURBATION}
    assert f6.NAME not in plugins["r5b_gravity_scale"].families


@pytest.mark.parametrize("magnitude", [0.05, 0.4])
def test_the_perturbation_scales_gravity_by_one_plus_or_minus_m(
    plugins: dict[str, Any], scenes: list[SceneSpec], magnitude: float
) -> None:
    scene = scenes[0]
    edited, record = plugins[PERTURBATION].apply(scene, magnitude, np.random.default_rng(1))
    k = record.params["gravity_scale"]
    assert k == pytest.approx(1.0 + magnitude) or k == pytest.approx(1.0 - magnitude)
    assert edited.gravity[2] == pytest.approx(k * scene.gravity[2])
    assert record.name == PERTURBATION
