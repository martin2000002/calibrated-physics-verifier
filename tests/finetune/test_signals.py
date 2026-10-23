from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from physverify.datasets.local_video import QueryEntry
from physverify.finetune.reward_hacking_signals import motion_index, objects_ok
from physverify.schema.states import ObjectStateTable

States = Callable[..., ObjectStateTable]


def test_a_still_video_has_zero_motion(make_states: States) -> None:
    position = np.zeros((5, 1, 3))
    assert motion_index(make_states(position, np.full((5, 1, 3), 0.5), ["ball"])) == 0.0


def test_motion_is_displacement_over_extent(make_states: States) -> None:
    position = np.zeros((4, 2, 3))
    position[:, 0, 0] = [0.0, 1.0, 3.0, 2.0]
    position[:, 1, 1] = [0.0, 0.0, 1.0, 1.0]
    size = np.full((4, 2, 3), 0.5)
    size[:, 1] = 2.0
    states = make_states(position, size, ["ball", "cube"])
    assert motion_index(states) == pytest.approx(6.0)


def test_tracking_jitter_does_not_accumulate_into_motion(make_states: States) -> None:
    position = np.zeros((101, 1, 3))
    position[1::2, 0, 0] = 0.01
    states = make_states(position, np.full((101, 1, 3), 0.5), ["ball"])
    assert motion_index(states) == pytest.approx(0.02)


def test_motion_is_none_without_finite_positions(make_states: States) -> None:
    states = make_states(np.full((3, 1, 3), np.nan), np.ones((3, 1, 3)), ["ball"])
    assert motion_index(states) is None


def test_objects_ok_counts_visible_objects_by_label(make_states: States) -> None:
    visible = np.ones((10, 3), dtype=bool)
    visible[:3, 2] = False
    states = make_states(np.zeros((10, 3, 3)), np.ones((10, 3, 3)), ["cube"] * 3, visible)
    assert objects_ok(states, [QueryEntry(phrase="cube", expected_count=2)]) is True
    assert objects_ok(states, [QueryEntry(phrase="cube", expected_count=3)]) is False
    assert objects_ok(states, [QueryEntry(phrase="ball", expected_count=1)]) is False
    assert objects_ok(states, ["cube"]) is None
