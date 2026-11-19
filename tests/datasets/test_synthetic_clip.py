from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import synthetic_clip as fx

from physverify.config.loader import load_settings
from physverify.config.models import RuntimeSettings
from physverify.datasets._base import DatasetSource, GroundTruthStore, VideoSource
from physverify.extraction._base import GROUND_TRUTH, StageContext
from physverify.physics.contact_manifold import support_polygon
from physverify.physics.finite_diff import second_difference
from physverify.physics.geometry import Manifold, characteristic_length
from physverify.physics.measures.gravity_fit import fall_fit, gravity_rel_error
from physverify.physics.measures.support_overhang import overhang_fraction
from physverify.schema.clip import SamplingSpec, resolve_sampling
from physverify.schema.perception import DepthKind, IntrinsicsSource
from physverify.schema.scale import ScaleEstimate, ScaleSource, effective_scale
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame


@pytest.fixture(scope="module")
def clip() -> fx.SyntheticClip:
    return fx.build_clip(RuntimeSettings())


def test_protocols_hold_at_runtime(clip: fx.SyntheticClip) -> None:
    dataset = fx.SyntheticFixtureDataset(settings=None)
    assert isinstance(dataset, DatasetSource)
    assert isinstance(clip.video, VideoSource)
    assert isinstance(clip.ground_truth, GroundTruthStore)
    assert dataset.clips()[0] == clip.manifest


def test_resampled_ground_truth_follows_the_sampling(clip: fx.SyntheticClip) -> None:
    sampling = resolve_sampling(clip.manifest, SamplingSpec(target_fps=12.0, max_side=48))
    assert (sampling.width, sampling.height, sampling.step) == (48, 36, 2)
    frames = clip.video.frames(sampling.start, sampling.stop, sampling.step, sampling.max_side)
    masks = clip.ground_truth.masks(sampling)
    depth = clip.ground_truth.depth(sampling)
    camera = clip.ground_truth.camera(sampling)
    assert frames.shape == (6, 36, 48, 3)
    assert masks.masks.shape == (6, 2, 36, 48) and bool(masks.masks.any(axis=(2, 3)).all())
    assert depth.kind is DepthKind.METRIC_Z and depth.depth.shape == (6, 36, 48)
    assert camera.source is IntrinsicsSource.GROUND_TRUTH
    np.testing.assert_allclose(camera.K[:2], fx.intrinsics()[:2] / 2.0, atol=1e-12)


def test_detections_contain_the_projected_ball(clip: fx.SyntheticClip) -> None:
    sampling = resolve_sampling(clip.manifest, SamplingSpec())
    detections = clip.ground_truth.detections(sampling)
    ball = list(detections.object_ids).index("ball")
    x0, y0, x1, y1 = detections.boxes_xyxy[ball]
    k = fx.intrinsics()
    centre = fx.SPHERE_START - fx.CAMERA_POSITION
    u = k[0, 2] + k[0, 0] * centre[0] / centre[1]
    v = k[1, 2] - k[1, 1] * centre[2] / centre[1]
    assert x0 <= u <= x1 and y0 <= v <= y1


def test_depth_is_metric_along_the_optical_axis(clip: fx.SyntheticClip) -> None:
    sampling = resolve_sampling(clip.manifest, SamplingSpec())
    depth = clip.ground_truth.depth(sampling)
    near_ground = depth.depth[0, -1, 0]
    ray_v = (fx.HEIGHT - 0.5 - fx.HEIGHT / 2) / fx.FOCAL_PX
    assert near_ground == pytest.approx(fx.CAMERA_POSITION[2] / ray_v, rel=1e-6)
    assert not depth.valid[0, 0].any()


def test_ground_truth_kinematics_are_analytic(clip: fx.SyntheticClip) -> None:
    states = clip.ground_truth.states(CoordFrame.WORLD_ZUP)
    ball = states.position[:, 0]
    acceleration = second_difference(ball, states.t)[1:-1]
    np.testing.assert_allclose(acceleration, [[0.0, 0.0, -9.81]] * 10, atol=1e-9)
    fit = fall_fit(states.t, ball)
    assert gravity_rel_error(float(np.linalg.norm(fit.a)), 1.0) < 1e-9
    camera = clip.ground_truth.states(CoordFrame.CAMERA_OPENCV)
    np.testing.assert_allclose(camera.gravity_dir, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(camera.position[0, 1], [0.4, 0.35, 4.0], atol=1e-12)


def test_resting_box_is_inside_its_support(clip: fx.SyntheticClip) -> None:
    table = clip.ground_truth.contacts(CoordFrame.WORLD_ZUP)
    record = table.records[0]
    assert record.points.shape == (4, 3)
    assert record.point_distances is not None
    manifold = Manifold(record.points, np.tile(record.normal, (4, 1)), record.point_distances)
    polygon = support_polygon([manifold], fx.GRAVITY_DIR)
    overhang = overhang_fraction(fx.BOX_CENTER, polygon, fx.GRAVITY_DIR, fx.BOX_EXTENTS)
    assert overhang == pytest.approx(-0.2 / characteristic_length(fx.BOX_EXTENTS))


def test_ground_truth_scale_binds_to_the_state_table(clip: fx.SyntheticClip) -> None:
    states = clip.ground_truth.states(CoordFrame.WORLD_ZUP)
    estimate = ScaleEstimate(
        meters_per_unit=1.0, sigma=0.0, source=ScaleSource.GROUND_TRUTH, lift_key=states.source_key
    )
    scale = effective_scale(states, estimate)
    assert scale is not None and scale.rel_sigma == 0.0


def test_stage_context_gives_zero_error_for_ground_truth_upstream(
    clip: fx.SyntheticClip, config_dir: Path, environ: dict[str, str]
) -> None:
    settings = load_settings(config_dir, environ=environ)
    sampling = resolve_sampling(clip.manifest, SamplingSpec())
    context = StageContext(
        clip=clip.manifest,
        video=clip.video,
        sampling=sampling,
        upstream={},
        plan={Stage.DEPTH: GROUND_TRUTH, Stage.TRACK: "edgetam"},
        gt=clip.ground_truth,
        error_profile=None,
        device="cpu",
        threads=1,
        paths=settings.paths,
    )
    exact = context.error_entry(Stage.DEPTH, "abs_rel")
    assert exact is not None and exact.sigma == 0.0 and exact.bias == 0.0
    assert context.error_entry(Stage.TRACK, "mask_iou") is None
    assert context.require_gt() is clip.ground_truth
    assert math.isfinite(sampling.seconds_per_frame)
