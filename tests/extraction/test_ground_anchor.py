"""Ground-plane anchoring of the lift: geometry, floor-row selection, stack guard, fallbacks,
sigma, profile independence and v5 equality."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from support import entry, make_context, profile_of, run_provider
from synthetic_clip import SyntheticClip
from test_lift_backproject import lift, truth_inputs

from physverify.extraction._base import GROUND_TRUTH
from physverify.extraction.floor_plane import FloorSampling, plane_normal
from physverify.extraction.ground_anchor import (
    AnchorMode,
    AnchorPath,
    AnchorResult,
    GroundAnchorParams,
    anchor_clip,
    anchor_ratio_envelope,
    anchored_fields,
    anchored_sigma,
    elevation_gain,
    implied_height,
    interpolated_factor,
    mask_angular_radius,
    normal_usable,
    ray_plane_distance,
    row_noise,
    steady_rows,
)
from physverify.extraction.providers import lift_ground_anchored
from physverify.extraction.providers.lift_ground_anchored import GroundAnchoredLift
from physverify.extraction.stack_guard import stacked_rows
from physverify.physics.geometry import characteristic_lengths
from physverify.schema.error_profile import ErrorProfile
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable

PITCH = math.radians(30.0)
DOWN = np.array([0.0, math.cos(PITCH), math.sin(PITCH)])
CAMERA_HEIGHT = 2.0
RADIUS = 0.1
PARAMS = GroundAnchorParams(
    envelope_quantile=0.9,
    anchor_z=2.0,
    min_anchor_rows=5,
    max_normal_sigma_rad=math.radians(10),
    steady_window=7,
    min_elevation_rad=math.radians(10),
    anchor_mode=AnchorMode.PER_ROW,
)
PIXEL = 1e-3
N_T = 30


def resting_centre(direction: np.ndarray, lift_m: float = 0.0) -> np.ndarray:
    """Centre of a sphere of RADIUS whose lowest point is ``lift_m`` above the plane, on a ray."""
    ray = direction / np.linalg.norm(direction)
    return np.asarray((CAMERA_HEIGHT - RADIUS - lift_m) / float(ray @ DOWN) * ray)


def scene(bias: tuple[float, ...], heights: np.ndarray) -> tuple[np.ndarray, ...]:
    """Spheres seen through the network with a constant depth bias each; ``heights`` (T, N) is
    each lowest point above the plane."""
    n_obj = len(bias)
    truth = np.full((N_T, n_obj, 3), np.nan)
    for t in range(N_T):
        for n in range(n_obj):
            direction = np.array([-0.6 + 0.4 * n + 0.01 * t, 0.2, 1.0])
            truth[t, n] = resting_centre(direction, float(heights[t, n]))
    scale = np.asarray(bias)[None, :, None]
    size = np.full((N_T, n_obj, 3), 2 * RADIUS) * scale
    quaternion = np.full((N_T, n_obj, 4), np.nan)
    return truth, truth * scale, size, quaternion


def bouncing_heights() -> np.ndarray:
    heights = np.zeros((N_T, 3))
    heights[:12, 1] = np.linspace(0.8, 0.05, 12)
    heights[:, 2] = np.linspace(1.2, 0.3, N_T)
    return heights


def anchored(
    bias: tuple[float, ...] = (1.0, 1.3, 0.8),
    excluded: np.ndarray | None = None,
    params: GroundAnchorParams = PARAMS,
) -> tuple[np.ndarray, AnchorResult]:
    truth, position, size, quaternion = scene(bias, bouncing_heights())
    pixel = np.full((N_T, len(bias)), PIXEL)
    spheres = np.ones(len(bias), dtype=bool)
    excluded = np.zeros((N_T, len(bias)), dtype=bool) if excluded is None else excluded
    result = anchor_clip(position, size, quaternion, spheres, DOWN, 0.01, pixel, excluded, params)
    return truth, result


def test_ray_plane_distance_places_a_resting_sphere_on_the_plane() -> None:
    centre = resting_centre(np.array([0.3, 0.1, 1.0]))
    distance = float(np.linalg.norm(centre))
    ray = centre / distance
    found = ray_plane_distance(ray[None], np.array([RADIUS / distance]), DOWN, CAMERA_HEIGHT)
    assert found[0] == pytest.approx(distance, rel=1e-12)
    height = implied_height(centre[None], np.array([RADIUS]), DOWN)
    assert height[0] == pytest.approx(CAMERA_HEIGHT, rel=1e-12)
    assert bool(np.isnan(ray_plane_distance(-DOWN[None], np.zeros(1), DOWN, CAMERA_HEIGHT)).all())


def test_envelope_is_the_quantile_of_the_positive_finite_heights() -> None:
    heights = np.array([*np.arange(1.0, 11.0), np.nan, -1.0])
    assert anchor_ratio_envelope(heights, 0.9) == pytest.approx(np.quantile(np.arange(1, 11), 0.9))
    assert math.isnan(anchor_ratio_envelope(np.array([np.nan, 0.0]), 0.9))


def test_a_constant_object_bias_is_recovered_and_every_row_shares_one_scale() -> None:
    truth, result = anchored()
    assert result.paths[:2] == (AnchorPath.ANCHORED, AnchorPath.ANCHORED)
    assert result.bias[1] / result.bias[0] == pytest.approx(1.3, rel=1e-9)
    _, position, *_ = scene((1.0, 1.3, 0.8), bouncing_heights())
    ratio = (position * result.factor[..., None])[:, :2] / truth[:, :2]
    np.testing.assert_allclose(ratio, ratio[0, 0, 0], rtol=1e-9)
    assert result.height == pytest.approx(CAMERA_HEIGHT * ratio[0, 0, 0], rel=1e-9)


def test_rows_in_flight_are_not_anchored() -> None:
    _, result = anchored()
    assert bool(result.anchored[:, 0].all())
    resting = bouncing_heights()[:, 1] == 0.0
    assert not bool((result.anchored[:, 1] & ~resting).any())
    settled = np.arange(N_T) >= int(np.argmax(resting)) + PARAMS.steady_window // 2
    assert bool(result.anchored[settled, 1].all())


def test_excluded_and_flat_rows_are_never_anchored() -> None:
    excluded = np.zeros((N_T, 3), dtype=bool)
    excluded[:10, 0] = True
    _, result = anchored(excluded=excluded)
    np.testing.assert_array_equal(result.anchored[:, 0], ~excluded[:, 0])
    flat = replace(PARAMS, min_elevation_rad=math.radians(60))
    assert anchored(params=flat)[1].paths == (AnchorPath.CLIP_FALLBACK,) * 3


def test_the_smoothed_mode_matches_a_constant_bias_and_interpolates_in_time() -> None:
    _, per_row = anchored()
    _, smoothed = anchored(params=replace(PARAMS, anchor_mode=AnchorMode.PER_ROW_SMOOTHED))
    np.testing.assert_allclose(smoothed.factor[:, :2], per_row.factor[:, :2], rtol=1e-9)
    rows = np.zeros(7, dtype=bool)
    rows[[1, 5]] = True
    factor = interpolated_factor(np.array([9, 2.0, 9, 9, 9, 8.0, 9]), rows, 1)
    expected = np.exp(np.interp(np.arange(7), [1, 5], np.log([2.0, 8.0])))
    np.testing.assert_allclose(factor, expected, rtol=1e-12)


def test_the_row_noise_ignores_the_motion_and_a_slope_breaks_steadiness() -> None:
    times = np.arange(400, dtype=np.float64)
    noise = np.random.default_rng(1).normal(0.0, 0.01, times.size)
    curve = 0.5 - 1e-3 * times + 2e-6 * times**2
    assert row_noise(curve + noise, 7) == pytest.approx(0.01, rel=0.15)
    assert row_noise(curve, 7) == pytest.approx(0.0, abs=1e-9)
    ramp = np.r_[np.zeros(10), 0.05 * np.arange(10.0), np.full(10, 0.45)]
    ramp[0] = np.nan
    steady = steady_rows(ramp, 0.01, 2.0, 7)
    assert bool(steady[:7].all()) and bool(steady[-6:].all())
    assert not bool(steady[12:18].any())


def test_an_object_never_at_its_envelope_keeps_the_network_depth() -> None:
    _, result = anchored()
    assert result.paths[2] is AnchorPath.UNANCHORED
    assert result.bias[2] == 1.0
    np.testing.assert_array_equal(result.factor[:, 2], 1.0)
    assert not bool(result.anchored[:, 2].any())


@pytest.mark.parametrize(
    ("down", "normal_sigma"),
    [(DOWN, math.radians(11.0)), (DOWN, math.nan), (np.full(3, np.nan), 0.01)],
)
def test_an_unusable_normal_leaves_the_whole_clip_unchanged(
    down: np.ndarray, normal_sigma: float
) -> None:
    _, position, size, quaternion = scene((1.0, 1.3), np.zeros((N_T, 2)))
    pixel = np.full((N_T, 2), PIXEL)
    spheres = np.ones(2, dtype=bool)
    excluded = np.zeros((N_T, 2), dtype=bool)
    result = anchor_clip(
        position, size, quaternion, spheres, down, normal_sigma, pixel, excluded, PARAMS
    )
    assert result.paths == (AnchorPath.CLIP_FALLBACK,) * 2
    assert math.isnan(result.height)
    np.testing.assert_array_equal(result.factor, 1.0)
    assert not normal_usable(down, normal_sigma, PARAMS)
    assert normal_usable(DOWN, 0.01, PARAMS)


def test_no_anchored_object_leaves_the_whole_clip_unchanged() -> None:
    heights = np.column_stack([np.linspace(1.2, 0.3, N_T)] * 2)
    _, position, size, quaternion = scene((1.0, 1.3), heights)
    pixel = np.full((N_T, 2), PIXEL)
    spheres = np.ones(2, dtype=bool)
    excluded = np.zeros((N_T, 2), dtype=bool)
    result = anchor_clip(position, size, quaternion, spheres, DOWN, 0.01, pixel, excluded, PARAMS)
    assert result.paths == (AnchorPath.CLIP_FALLBACK,) * 2


def test_anchored_sigma_and_elevation_gain() -> None:
    sin_beta = 0.5
    position = np.array([[0.0, sin_beta, math.sqrt(1 - sin_beta**2)]]) * 4.0
    down = np.array([0.0, 1.0, 0.0])
    gain = elevation_gain(position, np.zeros(1), down)
    assert gain[0] == pytest.approx(math.sqrt(3.0))
    sigma = anchored_sigma(np.array([2.0]), 0.1, 0.05, np.array([0.01]))
    assert sigma[0] == pytest.approx(math.sqrt(0.2**2 + 0.05**2 + 0.02**2))


def test_anchored_fields_scale_by_the_factor_and_swap_the_depth_term() -> None:
    fields = {
        "position": np.array([[[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]]]),
        "position_sigma": np.ones((1, 2, 3)),
        "size": np.ones((1, 2, 3)),
        "size_sigma": np.full((1, 2, 3), 0.2),
    }
    result = AnchorResult(
        factor=np.array([[2.0, 0.5]]),
        anchored=np.array([[True, False]]),
        bias=np.ones(2),
        paths=(AnchorPath.ANCHORED, AnchorPath.ANCHORED),
        height=1.0,
        gain=np.ones((1, 2)),
    )
    out = anchored_fields(fields, result, np.full((1, 2), 0.3), 0.1, 0.0)
    np.testing.assert_allclose(out["position"][0, 1], [0.0, 0.0, 5.0])
    np.testing.assert_allclose(out["position_sigma"][0, 1], 0.5)
    np.testing.assert_allclose(out["position_sigma"][0, 0, 2], math.sqrt(4 + 400 * 0.08))
    np.testing.assert_allclose(out["position_sigma"][0, 0, :2], 2.0)
    np.testing.assert_allclose(out["size_sigma"][0, 0], math.sqrt(0.16 + 4 * 0.08))


def test_the_focal_term_is_counted_once_on_anchored_cells() -> None:
    fields = {
        "position": np.array([[[2.0, 0.0, 10.0]]]),
        "position_sigma": np.ones((1, 1, 3)),
        "size": np.ones((1, 1, 3)),
        "size_sigma": np.full((1, 1, 3), 0.5),
    }
    result = AnchorResult(
        factor=np.ones((1, 1)),
        anchored=np.ones((1, 1), dtype=bool),
        bias=np.ones(1),
        paths=(AnchorPath.ANCHORED,),
        height=1.0,
        gain=np.ones((1, 1)),
    )
    out = anchored_fields(fields, result, np.full((1, 1), 0.3), 0.1, 0.2)
    np.testing.assert_allclose(out["position_sigma"][0, 0], [math.sqrt(1.16), 1.0, 3.0])
    np.testing.assert_allclose(out["size_sigma"][0, 0], math.sqrt(0.29))


def test_the_mask_radius_does_not_depend_on_the_mask_resolution() -> None:
    yy, xx = np.mgrid[:200, :200]
    full = ((yy - 100) ** 2 + (xx - 100) ** 2 <= 40**2)[None, None]
    half = full[:, :, ::2, ::2]
    radius = mask_angular_radius(full, 500.0, 1.0)
    assert radius[0, 0] == pytest.approx(40.0 / 500.0, rel=0.01)
    assert mask_angular_radius(half, 500.0, 0.5)[0, 0] == pytest.approx(radius[0, 0], rel=0.01)


def test_the_floor_normal_points_down_from_the_camera() -> None:
    rng = np.random.default_rng(0)
    x, z = rng.uniform(-2, 2, 400), rng.uniform(1, 6, 400)
    points = np.column_stack([x, (CAMERA_HEIGHT - z * DOWN[2]) / DOWN[1], z])
    normal = plane_normal(points, rng, FloorSampling(ransac_iters=50))
    assert normal is not None
    np.testing.assert_allclose(normal, DOWN, atol=1e-9)


def test_a_body_resting_on_another_one_in_the_image_is_stacked() -> None:
    masks = np.zeros((2, 3, 60, 40), dtype=bool)
    masks[:, 0, 10:20, 15:25] = True
    masks[0, 1, 22:32, 15:25] = True
    masks[1, 1, 26:36, 15:25] = True
    masks[:, 2, 40:50, 15:25] = True
    position = np.tile([0.0, 0.0, 5.0], (2, 3, 1))
    intrinsics = np.array([[100.0, 0.0, 20.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]])
    down = np.array([0.0, 1.0, 0.0])
    stacked = stacked_rows(masks, position, intrinsics, down, np.ones(2), 4)
    np.testing.assert_array_equal(stacked, [[True, False, False], [False, False, False]])
    masks[:, 1, 36:40, 15:25] = True
    stacked = stacked_rows(masks, position, intrinsics, down, np.ones(2), 4)
    np.testing.assert_array_equal(stacked[:, 1:], [[True, False], [True, False]])


def ground_anchored(
    clip: SyntheticClip,
    plan: dict[Stage, str] | None = None,
    profile: ErrorProfile | None = None,
    **params: object,
) -> ObjectStateTable:
    names = {
        **dict.fromkeys(STAGE_ORDER, GROUND_TRUTH),
        Stage.LIFT: "ground_anchored",
        **(plan or {}),
    }
    ctx = make_context(clip, upstream=truth_inputs(clip), plan=names, profile=profile)
    output = run_provider(GroundAnchoredLift, ctx, **params)
    assert isinstance(output, ObjectStateTable)
    return output


def test_without_a_normal_the_table_is_the_backprojection_one(fixture_clip: SyntheticClip) -> None:
    v5 = lift(fixture_clip, truth_inputs(fixture_clip), gravity="none")
    anchored_table = ground_anchored(fixture_clip, gravity="none")
    for name in ("position", "position_sigma", "size", "size_sigma", "visible"):
        np.testing.assert_array_equal(getattr(anchored_table, name), getattr(v5, name))


def test_truth_inputs_stay_within_a_quarter_length(fixture_clip: SyntheticClip) -> None:
    v5 = lift(fixture_clip, truth_inputs(fixture_clip))
    table = ground_anchored(fixture_clip)
    length = characteristic_lengths(v5.size)
    moved = np.linalg.norm(table.position - v5.position, axis=2) / length
    assert float(np.nanmax(moved)) <= 0.25


def test_the_selection_does_not_depend_on_a_loaded_profile(
    fixture_clip: SyntheticClip, monkeypatch: pytest.MonkeyPatch
) -> None:
    results: list[AnchorResult] = []

    def recorded(*args: Any) -> AnchorResult:
        results.append(anchor_clip(*args))
        return results[-1]

    monkeypatch.setattr(lift_ground_anchored, "anchor_clip", recorded)
    plan = {Stage.TRACK: "edgetam"}
    profile = profile_of(entry(Stage.TRACK, "edgetam", "centroid_px_rel", 5.0))
    bare = ground_anchored(fixture_clip, plan, min_elevation_deg=0.0)
    profiled = ground_anchored(fixture_clip, plan, profile, min_elevation_deg=0.0)
    assert bool(results[0].anchored.any())
    np.testing.assert_array_equal(results[1].anchored, results[0].anchored)
    np.testing.assert_array_equal(profiled.position, bare.position)
    np.testing.assert_array_equal(profiled.size, bare.size)
