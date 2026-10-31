"""LIFT v5: upright pose and local size from one view, on ray-cast solids of known truth."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from support import entry, profile_of
from synthetic_clip import SyntheticClip
from test_lift_backproject import lift, truth_inputs
from upright_scenes import UprightView, render

from physverify.extraction.backprojection import SummaryOptions, object_summary
from physverify.extraction.lift_geometry import (
    LocalGeometry,
    footprint_yaw,
    ground_frame,
    local_geometry,
    minimum_area_rectangle,
    orientation_sigma,
)
from physverify.physics.geometry import matrix_from_quaternion
from physverify.schema.stages import Stage
from physverify.schema.states import PrimitiveKind

OPTIONS = SummaryOptions(
    erode_px=1, extent_percentiles=(2.0, 98.0), min_area_px=9, footprint_trim_mads=3.0
)
CUBE = np.full(3, 0.4)
SIZE_REL_TOL = 0.35
YAW_TOL_DEG = 1.0


def lifted(kind: PrimitiveKind, view: UprightView) -> LocalGeometry:
    frame = ground_frame(view.gravity)
    assert frame is not None
    summary = object_summary(view.mask, view.z, view.mask, view.intrinsics, OPTIONS, frame)
    assert summary is not None and summary.view is not None
    geometry = local_geometry(kind, summary.view, float(summary.centre[2]), frame)
    assert geometry is not None
    return geometry


def yaw_error_deg(geometry: LocalGeometry, view: UprightView, period_deg: float) -> float:
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
def test_a_turned_cube_gives_its_yaw_up_axis_and_size(yaw: float, elevation: float) -> None:
    view = render(PrimitiveKind.BOX, CUBE, yaw, elevation)
    geometry = lifted(PrimitiveKind.BOX, view)
    assert yaw_error_deg(geometry, view, 90.0) < YAW_TOL_DEG
    up = matrix_from_quaternion(geometry.quaternion)[:, 2]
    assert float(up @ view.rotation[:, 2]) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(geometry.size, CUBE, rtol=SIZE_REL_TOL)


@pytest.mark.parametrize(
    ("kind", "extents"),
    [
        (PrimitiveKind.CYLINDER, np.array([0.3, 0.3, 0.5])),
        (PrimitiveKind.SPHERE, np.full(3, 0.4)),
    ],
)
def test_round_solids_stand_upright_with_their_size(
    kind: PrimitiveKind, extents: np.ndarray
) -> None:
    view = render(kind, extents, 0.0, 25.0)
    geometry = lifted(kind, view)
    up = matrix_from_quaternion(geometry.quaternion)[:, 2]
    assert float(up @ view.rotation[:, 2]) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(geometry.size, extents, rtol=SIZE_REL_TOL)


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
