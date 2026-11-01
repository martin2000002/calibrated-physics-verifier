"""The running median shared by R4_size v3 and LIFT's temporal filter."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.physics.running_median import EdgeMode, running_median

WINDOW = 5
SHIFTED: EdgeMode = "shifted"
KEEP: EdgeMode = "keep"
SHRINK: EdgeMode = "shrink"
ROWS = 12


def series(base: float = 2.0, rows: int = ROWS) -> np.ndarray:
    return np.full(rows, base)


@pytest.mark.parametrize("rows", [(0,), (0, 1), (5,), (5, 6), (ROWS - 1,), (ROWS - 2, ROWS - 1)])
def test_impulses_of_one_and_two_rows_are_removed(rows: tuple[int, ...]) -> None:
    z = series()
    z[list(rows)] = 3.0
    np.testing.assert_array_equal(running_median(z, WINDOW, SHIFTED), series())


@pytest.mark.parametrize("start", [0, 4, ROWS - 3])
def test_a_plateau_of_three_rows_is_kept(start: int) -> None:
    z = series()
    z[start : start + 3] = 3.0
    np.testing.assert_array_equal(running_median(z, WINDOW, SHIFTED), z)


@pytest.mark.parametrize("step_at", [3, 6, ROWS - 3])
def test_steps_with_three_rows_on_each_side_are_unchanged(step_at: int) -> None:
    step = series()
    step[step_at:] = 3.0
    np.testing.assert_array_equal(running_median(step, WINDOW, SHIFTED), step)


@pytest.mark.parametrize("ramp", [np.linspace(1.0, 4.0, ROWS) ** 1.5, np.linspace(4.0, 1.0, ROWS)])
def test_monotone_series_keep_their_interior_and_flatten_two_rows_at_each_end(
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
def test_keep_edges_leave_monotone_series_identical(ramp: np.ndarray) -> None:
    np.testing.assert_array_equal(running_median(ramp, WINDOW, KEEP), ramp)


def test_keep_edges_remove_interior_impulses_only() -> None:
    z = series()
    z[[0, 6]] = 3.0
    out = running_median(z, WINDOW, KEEP)
    assert out[0] == 3.0 and out[6] == 2.0


def test_a_short_series_takes_its_whole_median_or_keeps_its_values() -> None:
    short = np.array([1.0, 5.0, 1.0, 1.0])
    np.testing.assert_array_equal(running_median(short, WINDOW, SHIFTED), np.ones(4))
    np.testing.assert_array_equal(running_median(short, WINDOW, KEEP), short)


def test_non_finite_rows_are_untouched_and_not_sampled() -> None:
    z = series()
    z[3], z[7] = math.nan, 9.0
    out = running_median(z, WINDOW, SHIFTED)
    assert math.isnan(out[3])
    np.testing.assert_array_equal(np.delete(out, 3), np.delete(series(), 3))
    z = series()
    z[5] = math.inf
    out = running_median(z, WINDOW, SHIFTED)
    assert out[5] == math.inf
    np.testing.assert_array_equal(np.delete(out, 5), np.delete(series(), 5))


def test_the_shifted_window_keeps_its_width_at_the_ends() -> None:
    values = np.array([5.0, 1.0, 2.0, 3.0, np.nan, 4.0])
    found = running_median(values, 3, SHIFTED)
    assert found[0] == pytest.approx(2.0) and found[1] == pytest.approx(2.0)
    assert np.isnan(found[4]) and found[5] == pytest.approx(3.5)


@pytest.mark.parametrize("ramp", [np.arange(ROWS, dtype=np.float64), 5.0 - 0.3 * np.arange(ROWS)])
def test_shrinking_edges_leave_monotone_series_identical(ramp: np.ndarray) -> None:
    np.testing.assert_array_equal(running_median(ramp, WINDOW, SHRINK), ramp)


def test_the_shrinking_window_stays_centred_on_the_finite_rows() -> None:
    values = np.array([np.nan, 1.0, 9.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    found = running_median(values, WINDOW, SHRINK)
    assert np.isnan(found[0]) and np.isnan(found[7])
    assert found[1] == pytest.approx(1.0) and found[6] == pytest.approx(5.0)
    assert found[2] == pytest.approx(2.0)
    assert found[3] == pytest.approx(3.0) and found[5] == pytest.approx(4.0)


def test_shrinking_edges_remove_impulses_away_from_the_ends() -> None:
    z = series()
    z[[1, 5, ROWS - 2]] = 3.0
    np.testing.assert_array_equal(running_median(z, WINDOW, SHRINK), series())
    assert np.isnan(running_median(np.array([math.nan, math.nan]), WINDOW, SHRINK)).all()
