"""A metric depth plus its focal gives a lift in metres, and scale/metric_depth keeps it there.

Synthetic pinhole: an 8x8 px square on a fronto-parallel plane at Z0 metres, focal F px, the
principal point at the image centre. Pixel centres are (u + 0.5, v + 0.5), so the square's
median column 40..47 sits at u = 44 and X = (44 - W/2) Z0 / F."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from settings_factory import SettingsFactory
from support import construct, entry, make_context, profile_of, run_provider
from synthetic_clip import SyntheticClip
from wiring import resolve_plan

from physverify.diagnosis.chain_variants import ladder
from physverify.extraction._base import GROUND_TRUTH, StageContext, StageProvider
from physverify.extraction.providers._geocalib import camera_model
from physverify.extraction.providers.camera_from_depth import DepthFocalCamera
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.extraction.providers.scale_metric_depth import MetricDepthScale
from physverify.schema.clip import FrameSampling
from physverify.schema.error_profile import FOCAL_REL, SCALE_REL, ErrorEntry, ErrorProfile
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import DepthKind, DepthMap, MaskTrack
from physverify.schema.scale import ScaleEstimate, ScaleSource, effective_scale
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import LengthUnit, ObjectStateTable, PrimitiveKind

W, H, N_T = 64, 48, 3
F_PX = 80.0
Z0_M = 2.5
COLS, ROWS = slice(40, 48), slice(10, 18)
SIDE_PX = 8
TOL_M = 1e-6
PLAN = {
    **dict.fromkeys(STAGE_ORDER, GROUND_TRUTH),
    Stage.DEPTH: "moge2",
    Stage.CAMERA: "from_depth",
    Stage.LIFT: "backproject",
    Stage.SCALE: "metric_depth",
}
METRIC_ARM = (
    "extraction.plan.depth=moge2",
    "extraction.plan.camera=from_depth",
    "extraction.plan.lift=backproject",
    "extraction.plan.scale=metric_depth",
    "extraction.providers.camera.from_depth.gravity=none",
)
EXACT_GEOMETRY = {
    "erode_px": 0,
    "extent_percentiles": (0.0, 100.0),
    "center_offset_frac": {PrimitiveKind.UNKNOWN: 0.0},
}


def pinhole_sampling() -> FrameSampling:
    return FrameSampling(
        frame_indices=np.arange(N_T, dtype=np.int64),
        width=W,
        height=H,
        native_width=W,
        native_height=H,
        seconds_per_frame=0.1,
        step=1,
        max_side=None,
    )


def plane_inputs(kind: DepthKind = DepthKind.METRIC_Z) -> dict[Stage, StageOutput]:
    sampling = pinhole_sampling()
    masks = np.zeros((N_T, 1, H, W), dtype=bool)
    masks[:, 0, ROWS, COLS] = True
    depth = DepthMap(
        sampling=sampling,
        depth=np.full((N_T, H, W), Z0_M, dtype=np.float32),
        kind=kind,
        valid=np.ones((N_T, H, W), dtype=bool),
        focal_px=F_PX,
    )
    track = MaskTrack(
        sampling=sampling,
        object_ids=np.array(["square"]),
        labels=np.array(["square"]),
        masks=masks,
        scores=np.ones((N_T, 1)),
    )
    camera = camera_model(sampling, F_PX, math.nan, None)
    return {Stage.TRACK: track, Stage.DEPTH: depth, Stage.CAMERA: camera}


def context(
    clip: SyntheticClip,
    upstream: dict[Stage, StageOutput],
    profile: ErrorProfile | None = None,
    plan: dict[Stage, str] = PLAN,
) -> StageContext:
    ctx = make_context(clip, upstream=upstream, plan=plan, profile=profile)
    return replace(ctx, sampling=pinhole_sampling())


def metric_scale(
    clip: SyntheticClip,
    table: ObjectStateTable,
    profile: ErrorProfile | None = None,
    plan: dict[Stage, str] = PLAN,
) -> ScaleEstimate:
    scale = run_provider(MetricDepthScale, context(clip, {Stage.LIFT: table}, profile, plan))
    assert isinstance(scale, ScaleEstimate)
    return scale


def metric_lift(clip: SyntheticClip) -> ObjectStateTable:
    table = run_provider(BackprojectLift, context(clip, plane_inputs()), **EXACT_GEOMETRY)
    assert isinstance(table, ObjectStateTable)
    return table


def test_the_lift_of_a_metric_depth_is_in_metres(fixture_clip: SyntheticClip) -> None:
    table = metric_lift(fixture_clip)
    assert table.unit is LengthUnit.METER
    expected = [(44.0 - W / 2) * Z0_M / F_PX, (14.0 - H / 2) * Z0_M / F_PX, Z0_M]
    np.testing.assert_allclose(table.position[:, 0], np.tile(expected, (N_T, 1)), atol=TOL_M)
    side_m = SIDE_PX * Z0_M / F_PX
    np.testing.assert_allclose(table.size[:, 0, :2], side_m, atol=TOL_M)


def test_metric_depth_scale_is_one_metre_per_unit(fixture_clip: SyntheticClip) -> None:
    table = metric_lift(fixture_clip)
    applied = effective_scale(table, metric_scale(fixture_clip, table))
    assert applied is not None
    assert applied.source is ScaleSource.METRIC_DEPTH
    assert applied.meters_per_unit == 1.0
    assert math.isnan(applied.rel_sigma)
    side_m = SIDE_PX * Z0_M / F_PX
    assert table.size[0, 0, 0] * applied.meters_per_unit == pytest.approx(side_m, abs=TOL_M)


def test_the_scale_sigma_is_the_depth_scale_error_alone(
    fixture_clip: SyntheticClip,
) -> None:
    table = metric_lift(fixture_clip)
    profile = profile_of(
        entry(Stage.DEPTH, "moge2", SCALE_REL, 0.12, bias=0.05),
        entry(Stage.DEPTH, "moge2", FOCAL_REL, 0.04, bias=-0.03),
    )
    applied = effective_scale(table, metric_scale(fixture_clip, table, profile))
    assert applied is not None
    assert applied.rel_sigma == pytest.approx(math.hypot(0.05, 0.12))


def test_on_true_depth_the_scale_is_exact(fixture_clip: SyntheticClip) -> None:
    table = metric_lift(fixture_clip)
    scale = metric_scale(fixture_clip, table, plan={**PLAN, Stage.DEPTH: GROUND_TRUTH})
    assert scale.source is ScaleSource.METRIC_DEPTH
    assert scale.sigma == 0.0


def test_a_relative_lift_has_no_scale(fixture_clip: SyntheticClip) -> None:
    upstream = plane_inputs(DepthKind.RELATIVE_DEPTH)
    table = run_provider(BackprojectLift, context(fixture_clip, upstream), **EXACT_GEOMETRY)
    assert isinstance(table, ObjectStateTable) and table.unit is LengthUnit.RELATIVE
    scale = metric_scale(fixture_clip, table)
    assert scale.source is ScaleSource.NONE
    assert scale.meters_per_unit is None
    assert effective_scale(table, scale) is None


def lift_entries() -> tuple[ErrorEntry, ...]:
    return (
        entry(Stage.DEPTH, "moge2", "depth_rel", 0.05),
        entry(Stage.LIFT, "backproject", "centroid_rel", 0.02),
        entry(Stage.LIFT, "backproject", "size_rel", 0.03),
    )


@pytest.mark.parametrize("focal", [(entry(Stage.DEPTH, "moge2", FOCAL_REL, 0.04),), ()])
def test_a_from_depth_camera_gives_the_lift_finite_sigma(
    fixture_clip: SyntheticClip, focal: tuple[ErrorEntry, ...]
) -> None:
    profile = profile_of(*lift_entries(), *focal)
    upstream = plane_inputs()
    camera = run_provider(
        DepthFocalCamera, context(fixture_clip, upstream, profile), gravity="none"
    )
    lift_ctx = context(fixture_clip, {**upstream, Stage.CAMERA: camera}, profile)
    table = run_provider(BackprojectLift, lift_ctx, **EXACT_GEOMETRY)
    assert isinstance(table, ObjectStateTable)
    assert np.isfinite(table.position_sigma).all()
    assert np.isfinite(table.size_sigma).all()


def test_at_v3_a_metric_depth_arm_keeps_an_exact_metric_scale(
    fixture_clip: SyntheticClip,
    make_settings: SettingsFactory,
    real_providers: dict[str, type[StageProvider]],
) -> None:
    v3 = ladder()[3]
    plan = resolve_plan(
        make_settings(*METRIC_ARM), real_providers, dataset="x", gt_stages=v3.gt_order
    )
    outputs: dict[Stage, StageOutput] = {}
    for step in plan.stages[: STAGE_ORDER.index(Stage.SCALE) + 1]:
        ctx = make_context(
            fixture_clip, upstream=outputs, plan=plan.names, gt=fixture_clip.ground_truth
        )
        outputs[step.stage] = construct(step.provider, step.params).run(ctx)
    lift, scale = outputs[Stage.LIFT], outputs[Stage.SCALE]
    assert isinstance(lift, ObjectStateTable) and lift.unit is LengthUnit.METER
    assert isinstance(scale, ScaleEstimate)
    assert scale.source is ScaleSource.METRIC_DEPTH
    assert scale.sigma == 0.0
