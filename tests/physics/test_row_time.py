"""Windows given in seconds: the row counts at each frame rate, and the former 24-fps counts."""

from __future__ import annotations

import pytest

from physverify.checks.r2_static_equilibrium import EquilibriumParams
from physverify.checks.r5a_fall_shape import FallShapeParams, FlightRows
from physverify.checks.r5b_gravity_value import GravityValueParams
from physverify.extraction.providers._impacts import ImpactRows
from physverify.extraction.providers._proximity import ProximityParams
from physverify.extraction.providers.contacts_kinematic import Params as KinematicParams
from physverify.extraction.providers.kinematics_savgol import Params as SavgolParams
from physverify.extraction.providers.scale_free_fall import Params as FreeFallParams
from physverify.physics.row_time import REFERENCE_FPS, odd_rows_in, reference_seconds, rows_in

REFERENCE_ROW = 1.0 / REFERENCE_FPS


@pytest.mark.parametrize("rows", range(1, 40))
def test_reference_seconds_give_back_their_rows_at_the_reference_rate(rows: int) -> None:
    assert rows_in(reference_seconds(rows), REFERENCE_ROW) == rows
    if rows % 2:
        assert odd_rows_in(reference_seconds(rows), REFERENCE_ROW) == rows


def test_minimums_floor_the_counts() -> None:
    assert rows_in(0.0, REFERENCE_ROW, minimum=0) == 0
    assert rows_in(0.01, REFERENCE_ROW, minimum=4) == 4
    assert odd_rows_in(0.01, REFERENCE_ROW, minimum=3) == 3
    assert odd_rows_in(0.01, REFERENCE_ROW, minimum=4) == 5


def test_defaults_are_the_former_row_counts_at_24_fps() -> None:
    savgol = SavgolParams()
    assert savgol.spec(REFERENCE_ROW).window == 7
    assert rows_in(ProximityParams().min_rest_s, REFERENCE_ROW) == 3
    assert ImpactRows.at(KinematicParams().side_s, REFERENCE_ROW) == ImpactRows(4, 3, 3)
    assert rows_in(FreeFallParams().min_fall_s, REFERENCE_ROW) == 5
    assert rows_in(EquilibriumParams().min_rest_s, REFERENCE_ROW) == 3
    assert FlightRows.of(FallShapeParams(), REFERENCE_ROW) == FlightRows(6, 4)
    assert rows_in(GravityValueParams().min_flight_s, REFERENCE_ROW) == 4


@pytest.mark.parametrize(("fps", "window"), [(16.0, 5), (24.0, 7), (30.0, 9), (120.0, 35)])
def test_the_savgol_window_follows_the_frame_rate(fps: float, window: int) -> None:
    spec = SavgolParams().spec(1.0 / fps)
    assert spec.window == window
    assert spec.window % 2 == 1 and spec.window > spec.polyorder


@pytest.mark.parametrize("fps", [2.0, 4.0, 8.0, 16.0])
def test_a_slow_clip_keeps_a_savgol_window_above_the_polynomial_order(fps: float) -> None:
    spec = SavgolParams(polyorder=4).spec(1.0 / fps)
    assert spec.window == 5


@pytest.mark.parametrize(
    ("fps", "rows"),
    [(16.0, ImpactRows(3, 3, 2)), (30.0, ImpactRows(5, 4, 4)), (120.0, ImpactRows(20, 15, 15))],
)
def test_impact_windows_follow_the_frame_rate(fps: float, rows: ImpactRows) -> None:
    assert ImpactRows.at(KinematicParams().side_s, 1.0 / fps) == rows


@pytest.mark.parametrize(
    ("fps", "rows"),
    [(16.0, FlightRows(4, 3)), (30.0, FlightRows(8, 5)), (120.0, FlightRows(30, 20))],
)
def test_flight_windows_follow_the_frame_rate(fps: float, rows: FlightRows) -> None:
    assert FlightRows.of(FallShapeParams(), 1.0 / fps) == rows
