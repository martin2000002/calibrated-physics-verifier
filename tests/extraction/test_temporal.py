"""it2 T9-T12: LIFT version 3, the running median of depth and image extent per visible run."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
from lift_golden_case import CASES, GOLDEN, table
from support import make_context, run_provider
from synthetic_clip import SyntheticClip
from test_lift_backproject import gaussian_entry

from physverify.core.cache_keys import params_hash
from physverify.extraction._base import GROUND_TRUTH
from physverify.extraction.providers._lifting import (
    LiftErrors,
    ObjectSummary,
    closed_form_sigma,
    monte_carlo_sigma,
    object_position,
)
from physverify.extraction.providers._temporal import (
    running_median,
    stabilize_run,
    stabilize_summaries,
)
from physverify.extraction.providers.lift_backproject import BackprojectLift
from physverify.schema.clip import FrameSampling
from physverify.schema.outputs import StageOutput
from physverify.schema.perception import (
    CameraModel,
    DepthKind,
    DepthMap,
    IntrinsicsSource,
    MaskTrack,
)
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.schema.states import ObjectStateTable

WINDOW = 5
SHIFTED = "shifted"
ROWS = 12
TOL = 1e-12


def series(base: float = 2.0, rows: int = ROWS) -> np.ndarray:
    return np.full(rows, base)


@pytest.mark.parametrize("rows", [(0,), (0, 1), (5,), (5, 6), (ROWS - 1,), (ROWS - 2, ROWS - 1)])
def test_t9_impulses_of_one_and_two_rows_are_removed(rows: tuple[int, ...]) -> None:
    z = series()
    z[list(rows)] = 3.0
    np.testing.assert_array_equal(running_median(z, WINDOW, SHIFTED), series())


@pytest.mark.parametrize("start", [0, 4, ROWS - 3])
def test_t9_a_plateau_of_three_rows_is_kept(start: int) -> None:
    z = series()
    z[start : start + 3] = 3.0
    np.testing.assert_array_equal(running_median(z, WINDOW, SHIFTED), z)


@pytest.mark.parametrize("step_at", [3, 6, ROWS - 3])
def test_t9_steps_with_three_rows_on_each_side_are_unchanged(step_at: int) -> None:
    step = series()
    step[step_at:] = 3.0
    np.testing.assert_array_equal(running_median(step, WINDOW, SHIFTED), step)


@pytest.mark.parametrize("ramp", [np.linspace(1.0, 4.0, ROWS) ** 1.5, np.linspace(4.0, 1.0, ROWS)])
def test_t9_monotone_series_keep_their_interior_and_flatten_two_rows_at_each_end(
    ramp: np.ndarray,
) -> None:
    """A border impulse of up to two rows is itself a monotone series, so the shifted window
    cannot keep both: the first and last two rows take the value of the third from that end."""
    out = running_median(ramp, WINDOW, SHIFTED)
    np.testing.assert_array_equal(out[2:-2], ramp[2:-2])
    np.testing.assert_array_equal(out[:2], [ramp[2], ramp[2]])
    np.testing.assert_array_equal(out[-2:], [ramp[-3], ramp[-3]])


@pytest.mark.parametrize(
    "ramp", [np.linspace(1.0, 4.0, ROWS) ** 1.5, np.linspace(4.0, 1.0, ROWS), np.full(ROWS, 2.0)]
)
def test_t9_keep_edges_leave_monotone_series_identical(ramp: np.ndarray) -> None:
    np.testing.assert_array_equal(running_median(ramp, WINDOW, "keep"), ramp)


def test_t9_keep_edges_remove_interior_impulses_only() -> None:
    z = series()
    z[[0, 6]] = 3.0
    out = running_median(z, WINDOW, "keep")
    assert out[0] == 3.0 and out[6] == 2.0


def test_t10_short_series_and_non_finite_rows_are_untouched() -> None:
    short = np.array([1.0, 5.0, 1.0, 1.0])
    np.testing.assert_array_equal(running_median(short, WINDOW, SHIFTED), short)
    z = series()
    z[3], z[7] = math.nan, 9.0
    out = running_median(z, WINDOW, SHIFTED)
    assert math.isnan(out[3])
    np.testing.assert_array_equal(np.delete(out, 3), np.delete(series(), 3))


def summary(z: float, image_extent: float = 0.05) -> ObjectSummary:
    centre = np.array([0.3 * z, -0.1 * z, z])
    lateral = np.array([image_extent, 1.5 * image_extent]) * z
    return ObjectSummary(centre, np.array([*lateral, lateral.mean()]), 6.0, 40)


def test_t10_runs_shorter_than_the_window_and_hidden_rows_are_untouched() -> None:
    depths = [2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 2.0, 2.0]
    column: list[ObjectSummary | None] = [summary(z) for z in depths]
    column[2] = None  # runs of 2 and 5 rows; only the second one is stabilised
    column[0] = summary(4.0)
    out = stabilize_summaries(column, WINDOW, SHIFTED)
    assert out[2] is None
    assert out[0] is column[0] and out[1] is column[1]
    assert out[4] is not None
    np.testing.assert_allclose(out[4].centre, summary(2.0).centre, atol=TOL)
    np.testing.assert_allclose(out[4].extents, summary(2.0).extents, atol=TOL)
    assert out[4].r_eq_px == column[4].r_eq_px  # type: ignore[union-attr]


def test_t10_an_image_extent_impulse_is_removed_and_e_z_follows_the_rule() -> None:
    column: list[ObjectSummary | None] = [summary(2.0 + 0.1 * t) for t in range(8)]
    column[3] = summary(2.3, image_extent=0.08)
    out = stabilize_summaries(column, WINDOW, SHIFTED)
    clean = summary(2.3)
    assert out[3] is not None
    np.testing.assert_allclose(out[3].extents, clean.extents, rtol=TOL)
    assert out[3].extents[2] == pytest.approx(out[3].extents[:2].mean(), rel=TOL)
    np.testing.assert_allclose(out[3].centre, clean.centre, rtol=TOL)


def test_t11_a_constant_ratio_rescales_position_extents_and_sigma() -> None:
    """With q constant, z_new = c z multiplies the centre and the extents by c, and so position
    and both sigma modes (the same draws) by c: positions and sigma come from the summaries."""
    c = 1.37
    errors = LiftErrors(
        depth_rel=gaussian_entry(Stage.DEPTH, "vda_small", "depth_rel", 0.05),
        centroid_px_rel=gaussian_entry(Stage.TRACK, "edgetam", "centroid_px_rel", 0.1),
        centroid_rel=gaussian_entry(Stage.LIFT, "backproject", "centroid_rel", 0.05),
        size_rel=gaussian_entry(Stage.LIFT, "backproject", "size_rel", 0.05),
        phi=0.03,
    )
    k = np.array([[120.0, 0.0, 64.0], [0.0, 120.0, 64.0], [0.0, 0.0, 1.0]])
    base = summary(3.0)
    centres, extents = stabilize_run(
        np.stack([base.centre] * 6) * np.array([1, 1, 1, c, 1, 1])[:, None],
        np.stack([base.extents] * 6) * np.array([1, 1, 1, c, 1, 1])[:, None],
        WINDOW,
        "keep",
    )
    np.testing.assert_allclose(centres[3], base.centre, rtol=TOL)
    np.testing.assert_allclose(extents[3], base.extents, rtol=TOL)
    scaled = ObjectSummary(c * base.centre, c * base.extents, base.r_eq_px, base.n_pixels)
    for s, factor in ((base, 1.0), (scaled, c)):
        position = object_position(s.centre, s.extents, 0.354)
        np.testing.assert_allclose(
            position, factor * object_position(base.centre, base.extents, 0.354), rtol=TOL
        )
        closed = closed_form_sigma(position, s.extents, s.r_eq_px, k, errors)
        reference = closed_form_sigma(
            object_position(base.centre, base.extents, 0.354), base.extents, 6.0, k, errors
        )
        for got, want in zip(closed, reference, strict=True):
            np.testing.assert_allclose(got, factor * want, rtol=1e-9)
        sampled = monte_carlo_sigma(s, k, 0.354, errors, np.random.default_rng(3), 400)
        ref_mc = monte_carlo_sigma(base, k, 0.354, errors, np.random.default_rng(3), 400)
        for got, want in zip(sampled, ref_mc, strict=True):
            np.testing.assert_allclose(got, factor * want, rtol=1e-9)


def receding_disk(n_t: int = 10, size: int = 96) -> dict[Stage, StageOutput]:
    """A fronto-parallel disk of radius 0.1 receding from z = 2 to 3, metric depth, f = 300."""
    depths = np.linspace(2.0, 3.0, n_t)
    sampling = FrameSampling(
        frame_indices=np.arange(n_t, dtype=np.int64),
        width=size,
        height=size,
        native_width=size,
        native_height=size,
        seconds_per_frame=1.0 / 24.0,
        step=1,
        max_side=None,
    )
    rows, cols = np.mgrid[0:size, 0:size] + 0.5
    radius_px = 300.0 * 0.1 / depths
    disk = np.stack([(rows - size / 2) ** 2 + (cols - size / 2) ** 2 <= r**2 for r in radius_px])
    depth = np.where(disk, depths[:, None, None], 10.0).astype(np.float32)
    k = np.array([[300.0, 0.0, size / 2], [0.0, 300.0, size / 2], [0.0, 0.0, 1.0]])
    track = MaskTrack(
        sampling=sampling,
        object_ids=np.array(["disk"]),
        labels=np.array(["ball"]),
        masks=disk[:, None],
        scores=np.ones((n_t, 1)),
    )
    camera = CameraModel(
        sampling=sampling,
        K=k,
        source=IntrinsicsSource.GROUND_TRUTH,
        fov_deg=math.degrees(2.0 * math.atan(size / 600.0)),
        fov_sigma_deg=0.0,
    )
    return {
        Stage.TRACK: track,
        Stage.DEPTH: DepthMap(
            sampling=sampling, depth=depth, kind=DepthKind.METRIC_Z, valid=np.ones_like(disk)
        ),
        Stage.CAMERA: camera,
    }


def lift_on(
    clip: SyntheticClip, upstream: Mapping[Stage, StageOutput], **params: object
) -> ObjectStateTable:
    track = upstream[Stage.TRACK]
    assert isinstance(track, MaskTrack)
    names = {**dict.fromkeys(STAGE_ORDER, GROUND_TRUTH), Stage.LIFT: "backproject"}
    ctx = replace(make_context(clip, upstream=upstream, plan=names), sampling=track.sampling)
    out = run_provider(BackprojectLift, ctx, **params)
    assert isinstance(out, ObjectStateTable)
    return out


def test_t11_a_smooth_monotone_clip_is_unchanged_inside_the_run(
    fixture_clip: SyntheticClip,
) -> None:
    upstream = receding_disk()
    median = lift_on(fixture_clip, upstream, temporal_filter="median")
    none = lift_on(fixture_clip, upstream, temporal_filter="none")
    shifted = lift_on(fixture_clip, upstream, temporal_filter="median", temporal_edges=SHIFTED)
    for name in ("position", "position_sigma", "size", "size_sigma"):
        np.testing.assert_allclose(getattr(median, name), getattr(none, name), rtol=TOL)
        np.testing.assert_allclose(
            getattr(shifted, name)[2:-2], getattr(none, name)[2:-2], rtol=TOL
        )
    np.testing.assert_allclose(shifted.position[:2, 0, 2], none.position[2, 0, 2], rtol=TOL)
    np.testing.assert_allclose(shifted.position[-2:, 0, 2], none.position[-3, 0, 2], rtol=TOL)
    for name in ("visible", "in_frame", "occlusion", "confidence"):
        np.testing.assert_array_equal(getattr(median, name), getattr(none, name))


def test_t10_the_it1_gap_columns_do_not_depend_on_the_filter(fixture_clip: SyntheticClip) -> None:
    case = "noisy_monte_carlo"
    new = table(fixture_clip, case, temporal_filter="median")
    old = table(fixture_clip, case, temporal_filter="none")
    for name in ("visible", "in_frame", "occlusion", "confidence", "gravity_dir"):
        np.testing.assert_array_equal(getattr(new, name), getattr(old, name))
    assert not bool(np.array_equal(new.position, old.position, equal_nan=True))


@pytest.mark.parametrize("case", sorted(CASES))
def test_t12_none_reproduces_the_version_two_table(fixture_clip: SyntheticClip, case: str) -> None:
    golden = ObjectStateTable.load_npz(GOLDEN / f"{case}.npz")
    current = table(fixture_clip, case, temporal_filter="none")
    for name, value in golden.field_values().items():
        mine = getattr(current, name)
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            np.testing.assert_allclose(mine, value, rtol=TOL, atol=0.0, err_msg=name)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(mine, value, err_msg=name)
        else:
            assert mine == value or (mine != mine and value != value), name


def test_t12_the_version_and_both_parameters_enter_the_key() -> None:
    assert BackprojectLift.version == "3"
    default = BackprojectLift.Params()
    hashes = {
        params_hash(p.model_dump(mode="json"))
        for p in (
            default,
            BackprojectLift.Params(temporal_filter="median"),
            BackprojectLift.Params(temporal_window=7),
            BackprojectLift.Params(temporal_edges="shifted"),
        )
    }
    assert len(hashes) == 4
    assert default.temporal_filter == "none"  # type: ignore[attr-defined]
    assert default.temporal_edges == "keep"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="odd"):
        BackprojectLift.Params(temporal_window=6)
