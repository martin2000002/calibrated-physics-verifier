"""lift/smoothed_size: lengths re-referred to a running median of the object's surface depth."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from support import make_context, run_provider
from synthetic_clip import SyntheticClip
from test_lift_backproject import truth_inputs

from physverify.extraction._base import GROUND_TRUTH
from physverify.extraction.backprojection import ObjectSummary
from physverify.extraction.lift_geometry import SilhouetteView, ground_frame, local_geometry
from physverify.extraction.lift_params import LiftParams, SmoothedSizeLiftParams
from physverify.extraction.providers._backproject import BackprojectLift
from physverify.extraction.providers.lift_smoothed_size import (
    SmoothedSizeLift,
    rescaled,
    smoothed_sizes,
)
from physverify.physics.geometry import characteristic_lengths
from physverify.physics.running_median import EdgeMode
from physverify.schema.perception import DepthMap
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable, PrimitiveKind

VIEW = SilhouetteView(across=0.1, along=0.12, area=0.009, sin_elevation=0.3, yaw=0.4, ray_yaw=0.1)


def summary(depth: float) -> ObjectSummary:
    return ObjectSummary(
        centre=np.array([0.1, 0.2, depth]),
        extents=np.full(3, 0.1 * depth),
        r_eq_px=5.0,
        n_pixels=80,
        view=VIEW,
    )


def column(depths: list[float]) -> list[ObjectSummary | None]:
    return [None if math.isnan(z) else summary(z) for z in depths]


def ratios(
    depths: list[float],
    window: int = 5,
    edges: EdgeMode = "shifted",
    excluded: list[bool] | None = None,
) -> list[float]:
    """The factor each row's extents were multiplied by (NaN for a missing row)."""
    rows = column(depths)
    skip = np.array(excluded or [False] * len(rows), dtype=np.bool_)
    out = smoothed_sizes(rows, skip, window, edges)
    return [
        math.nan if before is None or after is None else after.extents[0] / before.extents[0]
        for before, after in zip(rows, out, strict=True)
    ]


def test_an_impulse_shorter_than_half_the_window_is_removed() -> None:
    depths = [2.0] * 9
    depths[4] = 2.5
    got = ratios(depths)
    assert got[4] == pytest.approx(2.0 / 2.5)
    assert [r for i, r in enumerate(got) if i != 4] == pytest.approx([1.0] * 8)


def test_a_step_and_a_monotone_run_are_kept_inside_the_series() -> None:
    assert ratios([2.0] * 5 + [3.0] * 5) == pytest.approx([1.0] * 10)
    assert ratios([2.0 + 0.1 * t for t in range(10)])[2:-2] == pytest.approx([1.0] * 6)


@pytest.mark.parametrize(("edges", "first"), [("shifted", 2.2 / 2.0), ("keep", 1.0)])
def test_edge_rows_use_a_window_shifted_inside_or_keep_their_size(
    edges: EdgeMode, first: float
) -> None:
    got = ratios([2.0 + 0.1 * t for t in range(10)], edges=edges)
    assert got[0] == pytest.approx(first)


def test_an_object_shorter_than_the_window_takes_the_median_of_all_its_rows() -> None:
    assert ratios([2.0, 2.4, 2.2], window=17) == pytest.approx([1.1, 2.2 / 2.4, 1.0])


def test_missing_and_excluded_rows_neither_enter_the_median_nor_change() -> None:
    depths = [2.0, math.nan, 2.0, 9.0, 2.0, 2.4, 2.0]
    excluded = [False, False, False, True, False, False, False]
    got = ratios(depths, excluded=excluded)
    assert math.isnan(got[1])
    assert got[3] == pytest.approx(1.0)
    assert got[5] == pytest.approx(2.0 / 2.4)
    assert smoothed_sizes([None, None], np.zeros(2, dtype=np.bool_), 5, "shifted") == [None, None]


def test_rescaling_moves_lengths_not_the_centre() -> None:
    before = summary(2.0)
    after = rescaled(before, 1.5)
    np.testing.assert_array_equal(after.centre, before.centre)
    np.testing.assert_allclose(after.extents, before.extents * 1.5)
    assert after.view is not None
    assert after.view.area == pytest.approx(VIEW.area * 1.5**2)
    assert (after.view.yaw, after.view.sin_elevation) == (VIEW.yaw, VIEW.sin_elevation)
    assert rescaled(replace(before, view=None), 1.5).view is None


@pytest.mark.parametrize(
    "primitive", [PrimitiveKind.SPHERE, PrimitiveKind.CYLINDER, PrimitiveKind.BOX]
)
def test_every_primitive_size_scales_with_the_ratio(primitive: PrimitiveKind) -> None:
    frame = ground_frame(np.array([0.0, 1.0, 0.0]))
    assert frame is not None
    plain = local_geometry(primitive, VIEW, 2.0, frame)
    scaled_view = rescaled(summary(2.0), 1.3).view
    assert plain is not None and scaled_view is not None
    scaled = local_geometry(primitive, scaled_view, 2.0, frame)
    assert scaled is not None
    np.testing.assert_allclose(scaled.size, plain.size * 1.3)
    np.testing.assert_allclose(scaled.offset_base, plain.offset_base * 1.3)


def test_the_window_must_be_odd() -> None:
    with pytest.raises(ValueError, match="depth_window must be odd"):
        SmoothedSizeLiftParams(depth_window=16)
    assert SmoothedSizeLiftParams().depth_window == 17


def noisy_lift(
    clip: SyntheticClip, provider: type[BackprojectLift], params: LiftParams
) -> ObjectStateTable:
    """The truth upstream with the depth of every fourth frame 20 % too far."""
    upstream = truth_inputs(clip)
    depth = upstream[Stage.DEPTH]
    assert isinstance(depth, DepthMap)
    factor = np.where(np.arange(depth.depth.shape[0]) % 4 == 1, 1.2, 1.0)
    upstream[Stage.DEPTH] = replace(
        depth, depth=(depth.depth * factor[:, None, None]).astype(np.float32)
    )
    names = {**dict.fromkeys(STAGE_ORDER, GROUND_TRUTH), Stage.LIFT: provider.name}
    output = run_provider(
        provider, make_context(clip, upstream=upstream, plan=names), **params.model_dump()
    )
    assert isinstance(output, ObjectStateTable)
    return output


def length_spread(table: ObjectStateTable) -> float:
    lengths = characteristic_lengths(np.asarray(table.size))
    rel = lengths / np.nanmedian(lengths, axis=0)
    return float(np.nanmedian(np.abs(rel - 1.0)))


def test_frame_depth_noise_leaves_the_lengths_but_not_the_centres(
    fixture_clip: SyntheticClip,
) -> None:
    window = 5
    plain = noisy_lift(fixture_clip, BackprojectLift, LiftParams())
    smooth = noisy_lift(fixture_clip, SmoothedSizeLift, SmoothedSizeLiftParams(depth_window=window))
    assert length_spread(smooth) < 0.25 * length_spread(plain)
    np.testing.assert_array_equal(smooth.visible, plain.visible)
    depth_ratio = smooth.position[..., 2] / plain.position[..., 2]
    assert np.nanmax(np.abs(depth_ratio - 1.0)) < 0.05
