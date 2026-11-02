"""Timing jitter of a state table: rows read at r + delta_r, nothing else changed."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from physverify.diagnosis.timing_jitter import (
    JitterSpec,
    apply_offsets,
    clip_rng,
    draw_offsets,
)
from physverify.schema.states import CoordFrame, LengthUnit, ObjectStateTable

FRAMES = 12


def table() -> ObjectStateTable:
    """Two objects moving along every axis at different speeds, sizes growing with the row."""
    blank = ObjectStateTable.blank(
        clip_id="clip",
        frame=CoordFrame.CAMERA_OPENCV,
        unit=LengthUnit.METER,
        source_key="lift-key",
        t=np.arange(FRAMES) / 24.0,
        frame_index=np.arange(FRAMES),
        object_ids=("a", "b"),
        labels=("cube", "ball"),
        primitive=("box", "sphere"),
    )
    rows = np.arange(FRAMES, dtype=np.float64)[:, None]
    position = np.stack([rows * [0.1, 0.2, 0.3], rows**2 * [-0.05, 0.0, 0.01] + 2.0], axis=1)
    size = np.stack([0.2 + rows * [0.01, 0.0, 0.02], 0.3 + rows * [0.0, 0.03, 0.0]], axis=1)
    return replace(blank, position=position, size=size)


def offsets_with(**values: float) -> np.ndarray:
    offsets = np.zeros(FRAMES)
    for row, delta in values.items():
        offsets[int(row.removeprefix("r"))] = delta
    return offsets


def test_a_fixed_seed_draws_the_same_offsets_per_clip() -> None:
    spec = JitterSpec(seed=7)
    first = draw_offsets(200, spec, clip_rng(spec, "clip_a"))
    assert np.array_equal(first, draw_offsets(200, spec, clip_rng(spec, "clip_a")))
    assert not np.array_equal(first, draw_offsets(200, spec, clip_rng(spec, "clip_b")))


def test_no_events_and_no_pervasive_jitter_is_the_identity() -> None:
    spec = JitterSpec(seed=7, event_rate=0.0, pervasive_sigma=0.0)
    offsets = draw_offsets(FRAMES, spec, clip_rng(spec, "clip"))
    assert not offsets.any()
    states = table()
    jittered = apply_offsets(states, offsets)
    assert np.array_equal(jittered.position, states.position)
    assert np.array_equal(jittered.size, states.size)


def test_a_repeated_frame_copies_the_previous_row_of_every_object() -> None:
    states = table()
    jittered = apply_offsets(states, offsets_with(r5=-1.0))
    assert np.array_equal(jittered.position[5], states.position[4])
    assert np.array_equal(jittered.size[5], states.size[4])
    others = np.arange(FRAMES) != 5
    assert np.array_equal(jittered.position[others], states.position[others])


@pytest.mark.parametrize("delta", [0.4, -0.3, 0.7])
def test_a_fractional_offset_interpolates_position_and_size(delta: float) -> None:
    states = table()
    jittered = apply_offsets(states, offsets_with(r6=delta))
    low = int(np.floor(6 + delta))
    a = 6 + delta - low
    for name in ("position", "size"):
        values = getattr(states, name)
        expected = (1 - a) * values[low] + a * values[low + 1]
        assert np.allclose(getattr(jittered, name)[6], expected, rtol=0.0, atol=1e-12)


def test_rows_with_an_unknown_neighbour_or_value_and_the_ends_stay() -> None:
    states = table()
    position = states.position.copy()
    position[3, 0, 1] = np.nan
    position[8, 1] = np.nan
    states = replace(states, position=position)
    offsets = offsets_with(r0=0.0, r4=-0.5, r8=0.5, r11=0.0)
    jittered = apply_offsets(states, offsets)
    assert np.array_equal(jittered.position[4, 0], states.position[4, 0])
    assert not np.array_equal(jittered.position[4, 1], states.position[4, 1])
    assert np.isnan(jittered.position[8, 1]).all()
    assert np.array_equal(jittered.position[[0, -1]], states.position[[0, -1]])
    drawn = draw_offsets(FRAMES, JitterSpec(seed=3, event_rate=1.0), np.random.default_rng(3))
    assert drawn[0] == drawn[-1] == 0.0


def test_the_input_table_is_not_changed() -> None:
    states = table()
    before = copy.deepcopy(states)
    spec = JitterSpec(seed=11, event_rate=0.5)
    apply_offsets(states, draw_offsets(FRAMES, spec, clip_rng(spec, "clip")))
    assert np.array_equal(states.position, before.position)
    assert np.array_equal(states.size, before.size)


def test_offsets_keep_every_row_within_the_table() -> None:
    spec = JitterSpec(seed=5, event_rate=1.0, pervasive_sigma=1.0)
    offsets = draw_offsets(500, spec, clip_rng(spec, "clip"))
    target = np.arange(500) + offsets
    assert target.min() >= 0.0 and target.max() <= 499.0


def test_the_event_rate_and_the_share_of_repeats_are_the_spec() -> None:
    spec = JitterSpec(seed=13, pervasive_sigma=0.0)
    offsets = draw_offsets(10_000, spec, clip_rng(spec, "clip"))
    events = offsets[offsets != 0.0]
    assert abs(events.size / 10_000 - 0.15) <= 0.01
    assert abs(np.mean(events == -1.0) - 0.59) <= 0.05
    halves = np.abs(events[events != -1.0])
    assert halves.min() >= 0.3 and halves.max() <= 0.7


def test_offsets_of_another_length_are_an_error() -> None:
    with pytest.raises(ValueError, match="offsets"):
        apply_offsets(table(), np.zeros(FRAMES + 1))
