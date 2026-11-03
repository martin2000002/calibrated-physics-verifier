"""Share of an object's mask box covered by the masks of the tracked objects nearer the camera."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.extraction.providers._visibility import nearer_cover

SIZE = 10


def rect(x0: int, x1: int, y0: int = 2, y1: int = 6) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def one_row(*masks: np.ndarray) -> np.ndarray:
    """(1, N, H, W) masks of a single row."""
    return np.stack(masks)[None]


def cover(masks: np.ndarray, depth: list[float], visible: list[bool] | None = None) -> np.ndarray:
    shown = [True] * len(depth) if visible is None else visible
    found: np.ndarray = nearer_cover(masks, np.array([depth]), np.array([shown]))
    return found[0]


SUBJECT, RIGHT, LEFT = rect(2, 6), rect(4, 8), rect(0, 4)


def test_only_a_nearer_mask_covers_the_box() -> None:
    # Subject box [2, 6) x [2, 6), 16 px: the nearer right mask covers columns 4-5 (8 px), the
    # farther left mask columns 2-3, which it does not count. The left object's box [0, 4) x
    # [2, 6) is covered on columns 2-3 by the subject (8 px); the right object is nearest.
    found = cover(one_row(SUBJECT, RIGHT, LEFT), [2.0, 1.0, 3.0])
    np.testing.assert_array_equal(found, [0.5, 0.0, 0.5])


def test_every_nearer_mask_counts_once() -> None:
    found = cover(one_row(SUBJECT, RIGHT, LEFT), [3.0, 1.0, 2.0])
    np.testing.assert_array_equal(found, [1.0, 0.0, 0.0])


def test_a_tie_in_depth_is_not_nearer() -> None:
    found = cover(one_row(SUBJECT, RIGHT), [2.0, 2.0])
    np.testing.assert_array_equal(found, [0.0, 0.0])


@pytest.mark.parametrize(
    ("depth", "visible"),
    [([2.0, math.nan], [True, True]), ([2.0, 1.0], [True, False]), ([2.0, math.inf], [True, True])],
)
def test_an_object_without_a_finite_visible_depth_is_unknown_and_covers_nothing(
    depth: list[float], visible: list[bool]
) -> None:
    found = cover(one_row(SUBJECT, RIGHT), depth, visible)
    assert found[0] == 0.0
    assert math.isnan(found[1])


def test_an_empty_mask_is_unknown() -> None:
    found = cover(one_row(np.zeros((SIZE, SIZE), dtype=bool), RIGHT), [2.0, 1.0])
    assert math.isnan(found[0])
    assert found[1] == 0.0


def test_the_box_not_the_silhouette_is_covered() -> None:
    # An L-shaped subject: its box [2, 6) x [2, 6) has a hole on columns 4-5, rows 4-5, which
    # the nearer mask fills without touching a subject pixel: 4 of 16 box pixels.
    subject = rect(2, 6, 2, 4) | rect(2, 4, 4, 6)
    found = cover(one_row(subject, rect(4, 6, 4, 6)), [2.0, 1.0])
    np.testing.assert_array_equal(found, [0.25, 0.0])


def test_rows_are_independent() -> None:
    masks = np.concatenate([one_row(SUBJECT, RIGHT), one_row(SUBJECT, RIGHT)])
    found = nearer_cover(masks, np.array([[2.0, 1.0], [1.0, 2.0]]), np.ones((2, 2), dtype=bool))
    np.testing.assert_array_equal(found, [[0.5, 0.0], [0.0, 0.5]])
