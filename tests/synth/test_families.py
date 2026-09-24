"""Every scene family samples valid, framed, seeded scenes from the committed settings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from physverify.schema.scene import SceneSpec
from physverify.synth.generate import base_scene, split_of
from physverify.synth.layout import (
    primitive_volume,
    project,
    rest_height,
    spec_intrinsics,
    spec_world_from_camera,
)
from physverify.synth.perturbations._measure import overhang_on, penetration_between
from physverify.synth.plan import ScenePlan, scene_id_of
from physverify.synth.scene_kit import solid_of
from physverify.synth.seeds import scene_rng
from physverify.synth.settings import DatasetSettings, family_params

FAMILIES = ("f1_projectile", "f2_bounce", "f3_collision", "f4_stack", "f5_contact")
INDICES = range(6)


def _scene(
    ds: DatasetSettings, families: Any, priors: dict[str, Any], family: str, i: int
) -> SceneSpec:
    plan = ScenePlan(scene_id_of(family, i), family, i, split_of(ds, family, i))
    return base_scene(plan, families[family], ds, priors)


@pytest.fixture(scope="module")
def sampled() -> dict[str, list[SceneSpec]]:
    """Scenes per family, drawn once per module (the settings are the same in every test)."""
    return {}


@pytest.fixture(params=FAMILIES)
def samples(
    request: pytest.FixtureRequest,
    sampled: dict[str, list[SceneSpec]],
    ds: DatasetSettings,
    families: Any,
    priors: dict[str, Any],
) -> list[SceneSpec]:
    family = request.param
    if family not in sampled:
        sampled[family] = [_scene(ds, families, priors, family, i) for i in INDICES]
    return sampled[family]


def test_families_are_registered(families: Any) -> None:
    assert set(FAMILIES) <= set(families)


def test_scenes_are_seeded_and_distinct(
    ds: DatasetSettings, families: Any, priors: dict[str, Any], samples: list[SceneSpec]
) -> None:
    family = samples[0].family
    assert _scene(ds, families, priors, family, 0) == samples[0]
    assert len({s.seed for s in samples}) == len(samples)
    assert len({s.objects[0].position for s in samples}) == len(samples)


def test_objects_are_known_sized_and_above_the_floor(
    samples: list[SceneSpec], priors: dict[str, Any], ds: DatasetSettings
) -> None:
    for spec in samples:
        assert (spec.n_frames, spec.frame_rate, spec.resolution) == (
            ds.n_frames,
            ds.fps,
            (ds.resolution, ds.resolution),
        )
        for obj in spec.objects:
            assert obj.label in priors
            assert min(obj.extents) > 0.0 and obj.mass > 0.0
            assert obj.position[2] >= rest_height(obj.extents) - 1e-6


def test_objects_start_apart(samples: list[SceneSpec]) -> None:
    for spec in samples:
        solids = [solid_of(o) for o in spec.objects]
        for i in range(len(solids)):
            for j in range(i + 1, len(solids)):
                assert penetration_between(solids[i], solids[j]) <= 1e-4


def test_objects_start_inside_the_image(samples: list[SceneSpec]) -> None:
    for spec in samples:
        centres = np.array([o.position for o in spec.objects])
        uv, z = project(
            centres,
            spec_intrinsics(spec.camera, spec.resolution),
            spec_world_from_camera(spec.camera),
        )
        width, height = spec.resolution
        assert np.all(z > 0.0)
        assert np.all((uv >= 0.0) & (uv <= (width, height)))


def density_of(obj: Any) -> float:
    return float(obj.mass / primitive_volume(obj.primitive, obj.extents))


def test_masses_are_density_times_volume(
    samples: list[SceneSpec], ds: DatasetSettings, families: Any, priors: dict[str, Any]
) -> None:
    params: Any = family_params(families[samples[0].family], ds, dict(priors))
    for scene in samples:
        densities = {o.object_id: density_of(o) for o in scene.objects}
        if scene.family == "f3_collision":
            # ball_1 has the family density; ball_0 is m_A = r m_B with r from mass_ratio_range.
            ratio = scene.objects[0].mass / scene.objects[1].mass
            assert params.mass_ratio_range[0] <= ratio <= params.mass_ratio_range[1]
            densities.pop("ball_0")
        assert densities == pytest.approx(dict.fromkeys(densities, params.density), rel=1e-12)


@pytest.mark.parametrize("t_c", [0.30, 0.37, 0.45])
def test_f3_balls_first_touch_at_the_drawn_collision_time(
    ds: DatasetSettings, families: Any, priors: dict[str, Any], collide: Any, t_c: float
) -> None:
    family = families["f3_collision"]
    params: Any = family_params(family, ds, dict(priors))
    fixed = params.model_copy(update={"collision_time_range": (t_c, t_c)})
    for index in INDICES:
        scene = family().sample(scene_rng(ds.seed, "f3_collision", index), fixed)
        a, b = scene.objects
        hit = collide(a, b, np.asarray(scene.gravity), 0.0)
        assert hit is not None
        assert hit.t == pytest.approx(t_c, abs=1e-9)


@pytest.mark.parametrize("samples", ["f4_stack"], indirect=True)
def test_f4_stacked_bodies_overhang_within_the_drawn_range(
    samples: list[SceneSpec], ds: DatasetSettings, families: Any, priors: dict[str, Any]
) -> None:
    params: Any = family_params(families["f4_stack"], ds, dict(priors))
    low, high = params.overhang_range
    checked = 0
    for scene in samples:
        for obj in scene.objects:
            others = [solid_of(o) for o in scene.objects if o.object_id != obj.object_id]
            if obj.position[2] - rest_height(obj.extents) > 1e-6:
                value = overhang_on(solid_of(obj), others, ds.contacts.tol_m)
                assert low - 1e-6 <= value <= high + 1e-6
                checked += 1
    assert checked >= len(samples)
