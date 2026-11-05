"""f5 camera axis: "across" (default) keeps the committed scenes, "along" films the motion line."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from physverify.schema.io import dumps
from physverify.schema.scene import SceneSpec
from physverify.synth.family_toolkit import along_motion_azimuth, heading, image_plane_azimuth
from physverify.synth.generate import base_scene, split_of
from physverify.synth.plan import ScenePlan, scene_id_of
from physverify.synth.settings import DatasetSettings

FAMILY = "f5_contact"
INDICES = range(12)
# sha256 of the twelve default scenes as sampled on main 58a116d, before the axis existed.
ACROSS_DIGEST = "22091639b18a18d0aa1c016d9543e0ce56a0bfc3ff71b928bf0ca403dff3c3dd"
SPREAD_DEG = 15.0


def _scenes(ds: DatasetSettings, families: Any, priors: dict[str, Any]) -> list[SceneSpec]:
    return [
        base_scene(
            ScenePlan(scene_id_of(FAMILY, i), FAMILY, i, split_of(ds, FAMILY, i)),
            families[FAMILY],
            ds,
            priors,
        )
        for i in INDICES
    ]


def _along(ds: DatasetSettings) -> DatasetSettings:
    return ds.model_copy(update={"families": {**ds.families, FAMILY: {"camera_axis": "along"}}})


def _camera_angle_deg(scene: SceneSpec) -> float:
    """Angle between the ball's motion and the horizontal camera-to-target line."""
    ball = next(o for o in scene.objects if o.object_id == "ball_0")
    motion = np.asarray(ball.velocity, dtype=np.float64)
    ray = np.asarray(scene.camera.look_at) - np.asarray(scene.camera.position)
    ray[2] = 0.0
    cosine = abs(float(motion @ ray)) / float(np.linalg.norm(motion) * np.linalg.norm(ray))
    return math.degrees(math.acos(min(1.0, cosine)))


def test_the_default_axis_keeps_the_committed_scenes(
    ds: DatasetSettings, families: Any, priors: dict[str, Any]
) -> None:
    digest = hashlib.sha256()
    for scene in _scenes(ds, families, priors):
        digest.update(dumps(scene.to_dict()).encode())
    assert digest.hexdigest() == ACROSS_DIGEST


def test_both_axes_draw_the_same_numbers() -> None:
    motion = heading(0.7)
    for seed in range(20):
        across, along = np.random.default_rng(seed), np.random.default_rng(seed)
        a = image_plane_azimuth(motion, across, SPREAD_DEG)
        b = along_motion_azimuth(motion, along, SPREAD_DEG)
        assert across.uniform() == along.uniform()
        assert math.isclose(abs(a - b) % 180.0, 90.0, abs_tol=1e-9)


def test_along_films_the_motion_line(
    ds: DatasetSettings, families: Any, priors: dict[str, Any]
) -> None:
    across = _scenes(ds, families, priors)
    along = _scenes(_along(ds), families, priors)
    for a, b in zip(across, along, strict=True):
        assert a.seed == b.seed and a.scene_id == b.scene_id
        assert b.camera != a.camera
        assert _camera_angle_deg(b) <= SPREAD_DEG + 1e-6
        assert _camera_angle_deg(a) >= 90.0 - SPREAD_DEG - 1e-6
