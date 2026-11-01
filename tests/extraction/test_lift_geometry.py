"""LIFT v5: upright pose and local size from one view, on ray-cast solids of known truth."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import binary_erosion
from support import entry, profile_of
from synthetic_clip import SyntheticClip
from test_lift_backproject import lift, truth_inputs
from upright_scenes import UprightView, render

from physverify.extraction import lift_geometry
from physverify.extraction.backprojection import (
    PlacedObject,
    SummaryOptions,
    object_position,
    object_summary,
    place_object,
)
from physverify.extraction.lift_geometry import (
    centre_depth_growth,
    cylinder_diameter,
    footprint_yaw,
    ground_frame,
    local_geometry,
    main_parts,
    minimum_area_rectangle,
    orientation_sigma,
)
from physverify.extraction.lift_params import LiftParams, summary_options
from physverify.physics.geometry import matrix_from_quaternion
from physverify.schema.stages import Stage
from physverify.schema.states import PrimitiveKind

PARAMS = LiftParams()
OPTIONS = summary_options(PARAMS)
CUBE = np.full(3, 0.4)
BOX_SIZE_REL_TOL = 0.12
ROUND_SIZE_REL_TOL = 0.05
BOX_CENTRE_REL_TOL = 0.16
ROUND_CENTRE_REL_TOL = 0.06
YAW_TOL_DEG = 1.0
DEPTH_NOISE = 0.005
BLEED_FRACTION = 0.3
BLEED_DEPTH_FACTOR = 1.6
NOISY_YAW_TOL_DEG = 2.0
UNTRIMMED_YAW_ERROR_DEG = 4.0


def lifted(
    kind: PrimitiveKind, view: UprightView, options: SummaryOptions = OPTIONS
) -> PlacedObject:
    frame = ground_frame(view.gravity)
    assert frame is not None
    summary = object_summary(view.mask, view.z, view.mask, view.intrinsics, options, frame, kind)
    assert summary is not None and summary.view is not None
    placed = place_object(summary, kind, PARAMS.center_offset_frac[kind], frame)
    assert placed.quaternion is not None
    return placed


def centre_error_rel(kind: PrimitiveKind, view: UprightView, extents: np.ndarray) -> float:
    """Distance from the lifted centre to the true one, relative to the largest extent."""
    frame = ground_frame(view.gravity)
    summary = object_summary(view.mask, view.z, view.mask, view.intrinsics, OPTIONS, frame, kind)
    assert summary is not None
    placed = place_object(summary, kind, PARAMS.center_offset_frac[kind], frame)
    position = object_position(summary.centre, placed.offset)
    return float(np.linalg.norm(position - view.centre)) / float(extents.max())


def yaw_error_deg(geometry: PlacedObject, view: UprightView, period_deg: float) -> float:
    assert geometry.quaternion is not None
    relative = view.rotation.T @ matrix_from_quaternion(geometry.quaternion)
    angle = math.degrees(math.atan2(relative[1, 0], relative[0, 0]))
    return abs((angle + period_deg / 2.0) % period_deg - period_deg / 2.0)


def test_the_smallest_rectangle_of_a_turned_rectangle_is_itself() -> None:
    angle = math.radians(25.0)
    grid = np.stack(np.meshgrid(np.linspace(0, 3, 13), np.linspace(0, 1, 5)), -1).reshape(-1, 2)
    turn = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    found = minimum_area_rectangle(grid @ turn.T + 7.0)
    assert found is not None
    assert found[0] == pytest.approx(angle, abs=1e-9)
    np.testing.assert_allclose(found[1], [3.0, 1.0], atol=1e-9)
    assert minimum_area_rectangle(np.column_stack([np.arange(5.0), np.arange(5.0)])) is None


@pytest.mark.parametrize("yaw", [0.0, 20.0, 45.0, 70.0])
@pytest.mark.parametrize("elevation", [15.0, 35.0])
def test_a_turned_cube_gives_its_yaw_up_axis_size_and_centre(yaw: float, elevation: float) -> None:
    view = render(PrimitiveKind.BOX, CUBE, yaw, elevation)
    geometry = lifted(PrimitiveKind.BOX, view)
    assert yaw_error_deg(geometry, view, 90.0) < YAW_TOL_DEG
    assert geometry.quaternion is not None
    up = matrix_from_quaternion(geometry.quaternion)[:, 2]
    assert float(up @ view.rotation[:, 2]) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(geometry.size, CUBE, rtol=BOX_SIZE_REL_TOL)
    assert centre_error_rel(PrimitiveKind.BOX, view, CUBE) < BOX_CENTRE_REL_TOL


@pytest.mark.parametrize(
    ("kind", "extents"),
    [
        (PrimitiveKind.CYLINDER, np.array([0.3, 0.3, 0.5])),
        (PrimitiveKind.SPHERE, np.full(3, 0.4)),
    ],
)
def test_round_solids_stand_upright_with_their_size_and_centre(
    kind: PrimitiveKind, extents: np.ndarray
) -> None:
    view = render(kind, extents, 0.0, 25.0)
    geometry = lifted(kind, view)
    assert geometry.quaternion is not None
    up = matrix_from_quaternion(geometry.quaternion)[:, 2]
    assert float(up @ view.rotation[:, 2]) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(geometry.size, extents, rtol=ROUND_SIZE_REL_TOL)
    assert centre_error_rel(kind, view, extents) < ROUND_CENTRE_REL_TOL


def test_a_sphere_keeps_its_diameter_on_the_camera_axes() -> None:
    view = render(PrimitiveKind.SPHERE, np.full(3, 0.4), 0.0, 25.0)
    summary = object_summary(
        view.mask, view.z, view.mask, view.intrinsics, OPTIONS, None, PrimitiveKind.SPHERE
    )
    assert summary is not None and summary.view is None
    placed = place_object(summary, PrimitiveKind.SPHERE, 0.354, None)
    assert placed.quaternion is None
    np.testing.assert_allclose(placed.size, 0.4, rtol=ROUND_SIZE_REL_TOL)


def test_a_stray_blob_does_not_stretch_the_extents() -> None:
    view = render(PrimitiveKind.BOX, CUBE, 0.0, 15.0)
    stray = view.mask.copy()
    stray[2:4, 2:4] = True
    assert int(main_parts(stray, PARAMS.min_part_frac).sum()) == int(view.mask.sum())
    clean, noisy = (
        object_summary(mask, view.z, view.mask, view.intrinsics, OPTIONS)
        for mask in (view.mask, stray)
    )
    assert clean is not None and noisy is not None
    np.testing.assert_array_equal(noisy.extents, clean.extents)


def test_a_cylinder_silhouette_area_gives_its_diameter() -> None:
    d, h, e = 0.3, 0.5, math.radians(30.0)
    along = h * math.cos(e) + d * math.sin(e)
    area = d * h * math.cos(e) + math.pi / 4.0 * d**2 * math.sin(e)
    assert cylinder_diameter(area, along, math.sin(e)) == pytest.approx(d, rel=1e-12)
    assert cylinder_diameter(d * h, h, 0.0) == pytest.approx(d, rel=1e-12)


def test_the_centre_depth_grows_with_the_offset_behind_the_surface() -> None:
    surface = np.array([0.0, 0.0, 2.0])
    growth = centre_depth_growth(surface, 0.5)
    assert 2.0 + 0.5 * growth == pytest.approx(2.0 * growth)
    assert centre_depth_growth(surface, 0.0) == 1.0
    assert centre_depth_growth(surface, 5.0) == 2.0


def test_without_gravity_or_box_yaw_there_is_no_pose() -> None:
    assert ground_frame(np.full(3, np.nan)) is None
    view = render(PrimitiveKind.BOX, CUBE, 30.0, 25.0)
    frame = ground_frame(view.gravity)
    assert frame is not None
    summary = object_summary(view.mask, view.z, view.mask, view.intrinsics, OPTIONS, frame)
    assert summary is not None and summary.view is not None
    few = footprint_yaw(np.zeros((3, 3)), frame, 3.0)
    assert math.isnan(few)
    blind = replace(summary.view, yaw=math.nan)
    assert local_geometry(PrimitiveKind.BOX, blind, 3.0, frame) is None
    assert local_geometry(PrimitiveKind.CYLINDER, blind, 3.0, frame) is not None


def test_orientation_sigma_is_unknown_without_a_measured_error() -> None:
    assert math.isnan(orientation_sigma(None, 0.0))
    own = entry(Stage.LIFT, "backproject", "orientation_rad", 0.03, bias=0.04)
    assert orientation_sigma(own, 0.0) == pytest.approx(0.05)
    assert orientation_sigma(own, 0.12) == pytest.approx(0.13)
    assert math.isnan(orientation_sigma(own, math.nan))


def test_the_lift_emits_a_pose_whose_sigma_needs_a_measured_error(
    fixture_clip: SyntheticClip,
) -> None:
    """The ball stands upright on every visible row; the box, seen 5 degrees from level, shows
    no top face, so its footprint is a line, its yaw unknown, and it keeps the version 4 form."""
    upstream = truth_inputs(fixture_clip)
    table = lift(fixture_clip, upstream)
    posed = np.isfinite(table.quaternion).all(axis=-1)
    assert not bool(posed[~table.visible].any())
    assert bool(posed[table.visible[:, 0], 0].all()) and not bool(posed[:, 1].any())
    assert bool(np.isnan(table.orientation_sigma).all())
    own = entry(Stage.LIFT, "backproject", "orientation_rad", 0.05)
    measured = lift(fixture_clip, upstream, profile=profile_of(own))
    assert np.allclose(measured.orientation_sigma[posed], 0.05)
    assert bool(np.isnan(measured.orientation_sigma[~posed]).all())
    np.testing.assert_array_equal(measured.quaternion, table.quaternion)


def test_without_gravity_the_lift_keeps_camera_axis_extents(fixture_clip: SyntheticClip) -> None:
    table = lift(fixture_clip, truth_inputs(fixture_clip), gravity="none")
    assert bool(np.isnan(table.quaternion).all())
    assert bool(np.isfinite(table.size[table.visible]).all())


def noisy(view: UprightView, rng: np.random.Generator) -> UprightView:
    """0.5 % depth noise, and edge pixels that bleed 60 % deeper (the background) at random."""
    z = view.z * (1.0 + DEPTH_NOISE * rng.standard_normal(view.z.shape))
    edge = view.mask & ~binary_erosion(view.mask, iterations=2)
    bleed = edge & (rng.random(view.z.shape) < BLEED_FRACTION)
    return replace(view, z=np.where(bleed, z * BLEED_DEPTH_FACTOR, z))


def test_trimming_the_footprint_keeps_the_yaw_under_depth_noise_and_edge_bleeding() -> None:
    rng = np.random.default_rng(0)
    untrimmed = replace(OPTIONS, footprint_trim_mads=1e9)
    for yaw in (10.0, 30.0, 60.0):
        view = noisy(render(PrimitiveKind.BOX, CUBE, yaw, 25.0), rng)
        trimmed_error = yaw_error_deg(lifted(PrimitiveKind.BOX, view), view, 90.0)
        raw_error = yaw_error_deg(lifted(PrimitiveKind.BOX, view, untrimmed), view, 90.0)
        assert trimmed_error < NOISY_YAW_TOL_DEG
        assert raw_error > UNTRIMMED_YAW_ERROR_DEG


def test_the_footprint_trim_reads_the_shared_robust_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    grid = np.stack(np.meshgrid(np.linspace(0, 4, 9), np.linspace(0, 1, 5)), -1).reshape(-1, 2)
    points = np.column_stack([np.vstack([grid, [[6.0, 6.0]]]), np.zeros(len(grid) + 1)])
    frame = lift_geometry.GroundFrame(np.eye(3))
    assert footprint_yaw(points, frame, 3.0) == pytest.approx(0.0, abs=1e-9)
    monkeypatch.setattr(lift_geometry, "robust_scale", lambda values: math.inf)
    assert footprint_yaw(points, frame, 3.0) > 1e-3
