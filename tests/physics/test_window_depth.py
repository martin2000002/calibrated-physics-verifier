"""Extents rescaled to the running-median depth of the object."""

from __future__ import annotations

import numpy as np

from physverify.physics.measures.window_depth import window_depth_factor

WINDOW = 5
ROWS = 12
DEPTH = 2.0


def depths(objects: int = 2) -> np.ndarray:
    return np.full((ROWS, objects), DEPTH)


def visible(objects: int = 2) -> np.ndarray:
    return np.ones((ROWS, objects), dtype=np.bool_)


def test_a_constant_depth_keeps_every_extent() -> None:
    np.testing.assert_array_equal(window_depth_factor(depths(), visible(), WINDOW), 1.0)


def test_a_depth_spike_is_taken_back_to_the_running_median() -> None:
    z = depths()
    z[[3, 7], 0] = [3.0, 1.0]
    factor = window_depth_factor(z, visible(), WINDOW)
    np.testing.assert_array_equal(factor[:, 0] * z[:, 0], DEPTH)
    np.testing.assert_array_equal(factor[:, 1], 1.0)


def test_rows_without_a_usable_depth_keep_their_extents() -> None:
    z = depths(1)
    z[[1, 2], 0] = [np.nan, 0.0]
    hidden = visible(1)
    hidden[5, 0] = False
    z[5, 0] = 9.0
    factor = window_depth_factor(z, hidden, WINDOW)
    np.testing.assert_array_equal(factor, 1.0)


def test_an_invisible_row_does_not_enter_the_median() -> None:
    z = depths(1)
    z[4:7, 0] = 9.0
    hidden = visible(1)
    hidden[4:7, 0] = False
    z[8, 0] = 3.0
    factor = window_depth_factor(z, hidden, WINDOW)
    assert factor[8, 0] * z[8, 0] == DEPTH


def test_a_table_without_objects_has_no_factor() -> None:
    assert window_depth_factor(np.empty((ROWS, 0)), np.empty((ROWS, 0), bool), WINDOW).shape == (
        ROWS,
        0,
    )
