"""AC5 and AC6: back-projection geometry, depth conversion, gravity and LIFT sigma."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest
from fakes import affine_disparity
from support import entry, make_context, profile_of, run_provider
from synthetic_clip import SyntheticClip

from physverify.extraction._base import GROUND_TRUTH
from physverify.extraction.matching import align_ground_truth
from physverify.extraction.providers._gravity_pose import LEVEL_CAMERA_GRAVITY
from physverify.extraction.providers._lifting import (
    LiftErrors,
    ObjectSummary,
    closed_form_sigma,
    lift_depth,
    monte_carlo_sigma,
    normalize_by_object_median,
    object_position,
    resolve_disparity_shift,
)
from physverify.extraction.providers.camera_assumed_fov import AssumedFovCamera
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.physics.geometry import characteristic_lengths
from physverify.schema.clip import FrameSampling
from physverify.schema.error_profile import ErrorEntry, ErrorProfile
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import DepthKind, DepthMap, MaskTrack
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import LengthUnit, ObjectStateTable

MIN_AREA_PX = 20
PERCEIVED = {Stage.TRACK: "edgetam", Stage.DEPTH: "vda_small", Stage.LIFT: "backproject"}


def truth_inputs(clip: SyntheticClip, *, depth: DepthMap | None = None) -> dict[Stage, StageOutput]:
    s = make_context(clip).sampling
    gt = clip.ground_truth
    return {
        Stage.TRACK: gt.masks(s),
        Stage.DEPTH: depth or gt.depth(s),
        Stage.CAMERA: gt.camera(s),
    }


def lift(
    clip: SyntheticClip,
    upstream: Mapping[Stage, StageOutput],
    *,
    plan: Mapping[Stage, str] | None = None,
    profile: ErrorProfile | None = None,
    **params: object,
) -> ObjectStateTable:
    names = {**dict.fromkeys(STAGE_ORDER, GROUND_TRUTH), Stage.LIFT: "backproject", **(plan or {})}
    ctx = make_context(clip, upstream=upstream, plan=names, profile=profile)
    output = run_provider(BackprojectLift, ctx, **params)
    assert isinstance(output, ObjectStateTable)
    return output


def own_entries(centroid: float, size: float) -> tuple[ErrorEntry, ...]:
    return (
        entry(Stage.LIFT, "backproject", "centroid_rel", centroid),
        entry(Stage.LIFT, "backproject", "size_rel", size),
    )


def test_truth_inputs_give_positions_within_a_quarter_length(fixture_clip: SyntheticClip) -> None:
    upstream = truth_inputs(fixture_clip)
    table = lift(fixture_clip, upstream)
    track = upstream[Stage.TRACK]
    assert isinstance(track, MaskTrack)
    truth = align_ground_truth(fixture_clip.ground_truth, track, track.sampling).states
    length = characteristic_lengths(truth.size)
    usable = track.masks.sum(axis=(2, 3)) >= MIN_AREA_PX
    position_error = np.linalg.norm(table.position - truth.position, axis=2) / length
    length_error = np.abs(characteristic_lengths(table.size) / length - 1.0)
    assert table.unit is LengthUnit.METER
    assert bool(usable.all())
    assert float(position_error[usable].max()) <= 0.25
    assert float(length_error[usable].max()) <= 0.25
    np.testing.assert_array_equal(table.frame_index, track.sampling.frame_indices)


@pytest.mark.parametrize(
    ("kind", "unit"),
    [
        (DepthKind.METRIC_Z, LengthUnit.METER),
        (DepthKind.RELATIVE_DEPTH, LengthUnit.RELATIVE),
        (DepthKind.AFFINE_DISPARITY, LengthUnit.RELATIVE),
    ],
)
def test_unit_follows_the_depth_kind(
    fixture_clip: SyntheticClip, kind: DepthKind, unit: LengthUnit
) -> None:
    metric = truth_inputs(fixture_clip)[Stage.DEPTH]
    assert isinstance(metric, DepthMap)
    depth = {
        DepthKind.METRIC_Z: metric,
        DepthKind.RELATIVE_DEPTH: DepthMap(
            sampling=metric.sampling,
            depth=metric.depth * np.float32(0.5),
            kind=DepthKind.RELATIVE_DEPTH,
            valid=metric.valid,
        ),
        DepthKind.AFFINE_DISPARITY: affine_disparity(metric, 2.0, 0.0),
    }[kind]
    assert lift(fixture_clip, truth_inputs(fixture_clip, depth=depth)).unit is unit


def test_zero_shift_disparity_recovers_the_relative_truth(fixture_clip: SyntheticClip) -> None:
    upstream = truth_inputs(fixture_clip)
    metric, masks = upstream[Stage.DEPTH], upstream[Stage.TRACK]
    assert isinstance(metric, DepthMap) and isinstance(masks, MaskTrack)
    z, valid, unit = lift_depth(affine_disparity(metric, 3.0, 0.0), masks, "zero_shift")
    expected, _ = normalize_by_object_median(
        metric.depth.astype(np.float64), metric.valid, masks.masks.any(axis=1)
    )
    assert unit is LengthUnit.RELATIVE
    np.testing.assert_allclose(z[valid], expected[valid], rtol=1e-6)


def disk_clip(depths: np.ndarray, shift: float, size: int = 160) -> tuple[DepthMap, MaskTrack]:
    """A fronto-parallel disk of radius 0.1 at depth z(t) seen with f = 1000 px."""
    n_t = depths.shape[0]
    sampling = FrameSampling(
        frame_indices=np.arange(n_t, dtype=np.int64),
        width=size,
        height=size,
        native_width=size,
        native_height=size,
        seconds_per_frame=0.1,
        step=1,
        max_side=None,
    )
    rows, cols = np.mgrid[0:size, 0:size] + 0.5
    radius_px = 1000.0 * 0.1 / depths
    masks = np.stack([(rows - size / 2) ** 2 + (cols - size / 2) ** 2 <= r**2 for r in radius_px])[
        :, None
    ]
    disparity = np.where(masks[:, 0], 2.0 / depths[:, None, None] + shift, shift)
    depth = DepthMap(
        sampling=sampling,
        depth=disparity.astype(np.float32),
        kind=DepthKind.AFFINE_DISPARITY,
        valid=np.ones_like(masks[:, 0]),
    )
    track = MaskTrack(
        sampling=sampling,
        object_ids=np.array(["disk"]),
        labels=np.array(["disk"]),
        masks=masks,
        scores=np.ones((n_t, 1)),
    )
    return depth, track


def test_size_constancy_recovers_the_disparity_shift() -> None:
    depth, track = disk_clip(np.linspace(2.0, 4.0, 6), shift=0.4)
    assert resolve_disparity_shift(depth, track, "size_constancy") == pytest.approx(0.4, rel=0.01)


def test_size_constancy_falls_back_to_zero_without_size_change(
    fixture_clip: SyntheticClip,
) -> None:
    upstream = truth_inputs(fixture_clip)
    metric, masks = upstream[Stage.DEPTH], upstream[Stage.TRACK]
    assert isinstance(metric, DepthMap) and isinstance(masks, MaskTrack)
    disparity = affine_disparity(metric, 2.0, 0.3)
    assert resolve_disparity_shift(disparity, masks, "size_constancy") == 0.0
    still, still_track = disk_clip(np.full(4, 3.0), shift=0.4)
    assert resolve_disparity_shift(still, still_track, "size_constancy") == 0.0


def test_gravity_modes(fixture_clip: SyntheticClip) -> None:
    upstream = truth_inputs(fixture_clip)
    posed = lift(fixture_clip, upstream, gravity="camera_pose")
    np.testing.assert_allclose(posed.gravity_dir, [0.0, 1.0, 0.0], atol=1e-9)
    assert posed.gravity_dir_sigma == 0.0
    assert bool(np.isnan(lift(fixture_clip, upstream, gravity="none").gravity_dir).all())
    auto = lift(fixture_clip, upstream)
    np.testing.assert_allclose(auto.gravity_dir, posed.gravity_dir, atol=1e-12)


def test_auto_gravity_without_pose_assumes_a_level_camera(fixture_clip: SyntheticClip) -> None:
    ctx = make_context(fixture_clip)
    camera = run_provider(AssumedFovCamera, ctx)
    upstream = {**truth_inputs(fixture_clip), Stage.CAMERA: camera}
    table = lift(fixture_clip, upstream, plan={Stage.CAMERA: "assumed_fov"})
    np.testing.assert_array_equal(table.gravity_dir, LEVEL_CAMERA_GRAVITY)
    assert table.gravity_dir_sigma == pytest.approx(math.radians(10.0))
    unknown = lift(
        fixture_clip, upstream, plan={Stage.CAMERA: "assumed_fov"}, gravity="camera_pose"
    )
    assert bool(np.isnan(unknown.gravity_dir).all())
    by_dataset = lift(
        fixture_clip,
        upstream,
        plan={Stage.CAMERA: "assumed_fov"},
        gravity_by_dataset={fixture_clip.manifest.dataset: "none"},
    )
    assert bool(np.isnan(by_dataset.gravity_dir).all())


def test_without_a_profile_perceived_inputs_leave_sigma_unknown(
    fixture_clip: SyntheticClip,
) -> None:
    table = lift(fixture_clip, truth_inputs(fixture_clip), plan=PERCEIVED)
    visible = table.visible
    assert bool(np.isnan(table.position_sigma[visible]).all())
    assert bool(np.isnan(table.size_sigma[visible]).all())


def test_truth_upstream_and_own_entry_give_exactly_ell_times_length(
    fixture_clip: SyntheticClip,
) -> None:
    profile = profile_of(*own_entries(centroid=0.1, size=0.05))
    table = lift(fixture_clip, truth_inputs(fixture_clip), profile=profile)
    visible = table.visible
    length = characteristic_lengths(table.size)
    expected = np.repeat((0.1 * length)[..., None], 3, axis=-1)
    np.testing.assert_allclose(table.position_sigma[visible], expected[visible], rtol=1e-12)
    np.testing.assert_allclose(table.size_sigma[visible], 0.05 * table.size[visible], rtol=1e-12)


def test_a_missing_read_entry_makes_the_lateral_sigma_unknown(
    fixture_clip: SyntheticClip,
) -> None:
    profile = profile_of(*own_entries(centroid=0.1, size=0.05))
    table = lift(
        fixture_clip, truth_inputs(fixture_clip), plan={Stage.TRACK: "edgetam"}, profile=profile
    )
    visible = table.visible
    assert bool(np.isnan(table.position_sigma[visible][:, :2]).all())
    assert bool(np.isfinite(table.position_sigma[visible][:, 2]).all())


def gaussian_entry(stage: Stage, provider: str, quantity: str, sigma: float) -> ErrorEntry:
    residuals = np.random.default_rng(len(quantity)).normal(0.0, sigma, 1000)
    return entry(
        stage,
        provider,
        quantity,
        float(residuals.std(ddof=1)),
        bias=float(residuals.mean()),
        n=residuals.size,
        residuals=tuple(float(r) for r in residuals),
    )


def test_closed_form_and_monte_carlo_agree_within_ten_percent(
    fixture_clip: SyntheticClip,
) -> None:
    profile = profile_of(
        gaussian_entry(Stage.DEPTH, "vda_small", "depth_rel", 0.05),
        gaussian_entry(Stage.TRACK, "edgetam", "centroid_px_rel", 0.1),
        gaussian_entry(Stage.LIFT, "backproject", "centroid_rel", 0.05),
        gaussian_entry(Stage.LIFT, "backproject", "size_rel", 0.05),
    )
    upstream = truth_inputs(fixture_clip)
    closed = lift(fixture_clip, upstream, plan=PERCEIVED, profile=profile)
    sampled = lift(
        fixture_clip,
        upstream,
        plan=PERCEIVED,
        profile=profile,
        sigma_mode="monte_carlo",
        mc_draws=4000,
    )
    visible = closed.visible
    for name in ("position_sigma", "size_sigma"):
        ratio = getattr(sampled, name)[visible] / getattr(closed, name)[visible]
        assert float(np.abs(ratio - 1.0).max()) < 0.10, name


@pytest.mark.parametrize("with_profile", [True, False])
@pytest.mark.parametrize("phi", [math.nan, 0.0, 0.02])
def test_an_unknown_focal_error_is_unknown_in_both_sigma_modes(
    phi: float, with_profile: bool
) -> None:
    def known(value: ErrorEntry) -> ErrorEntry | None:
        return value if with_profile else None

    errors = LiftErrors(
        depth_rel=known(gaussian_entry(Stage.DEPTH, "vda_small", "depth_rel", 0.05)),
        centroid_px_rel=known(gaussian_entry(Stage.TRACK, "edgetam", "centroid_px_rel", 0.1)),
        centroid_rel=known(gaussian_entry(Stage.LIFT, "backproject", "centroid_rel", 0.05)),
        size_rel=known(gaussian_entry(Stage.LIFT, "backproject", "size_rel", 0.05)),
        phi=phi,
    )
    summary = ObjectSummary(
        centre=np.array([0.4, -0.2, 3.0]), extents=np.full(3, 0.3), r_eq_px=12.0, n_pixels=450
    )
    intrinsics = np.array([[120.0, 0.0, 64.0], [0.0, 120.0, 64.0], [0.0, 0.0, 1.0]])
    position = object_position(summary.centre, summary.extents, 0.5)
    closed = closed_form_sigma(position, summary.extents, summary.r_eq_px, intrinsics, errors)
    rng = np.random.default_rng(0)
    sampled = monte_carlo_sigma(summary, intrinsics, 0.5, errors, rng, 2000)
    for cf, mc in zip(closed, sampled, strict=True):
        assert bool(np.isnan(mc[np.isnan(cf)]).all())
    unknown = math.isnan(phi) or not with_profile
    for sigma in (closed[0][:2], closed[1], sampled[0][:2], sampled[1]):
        assert bool(np.isnan(sigma).all()) is unknown
