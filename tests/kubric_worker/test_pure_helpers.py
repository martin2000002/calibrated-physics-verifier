"""Host-side checks of the container helpers that need neither Kubric nor Blender."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import records

from physverify.kubric_worker.build_scene import (
    HIDDEN_Z,
    PRIMITIVE_BOUNDS,
    clip_inputs,
    guarded,
    hidden_position,
    kubric_frame,
    kubric_scale,
    native_entries,
    probe_points,
    save_record,
    sim_frame_end,
)
from physverify.kubric_worker.tasks.contacts import (
    closest_arrays,
    object_pairs,
    present_objects,
    xyzw,
)
from physverify.kubric_worker.tasks.render import IDENTITY_WXYZ, pose_keyframes, to_uint8
from physverify.kubric_worker.tasks.simulate import raw_collisions, state_table
from physverify.schema.states import CoordFrame, ObjectStateTable


def _track(n: int, offset: float) -> dict[str, list[list[float]]]:
    return {
        "position": [[offset, 0.0, float(i)] for i in range(n)],
        "quaternion": [[1.0, 0.0, 0.0, 0.0]] * n,
        "velocity": [[0.0, 0.0, 1.0]] * n,
        "angular_velocity": [[0.0, 0.0, 0.0]] * n,
    }


def _settled_scene(settle: int) -> tuple[object, int]:
    spec = dataclasses.replace(records.scene(), settle_frames=settle)
    return spec, sim_frame_end(settle, spec.n_frames)


def test_kubric_scale_maps_the_primitive_span_to_full_extents() -> None:
    assert kubric_scale((0.4, 0.2, 0.1), *PRIMITIVE_BOUNDS) == pytest.approx((0.2, 0.1, 0.05))
    assert kubric_scale((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.5, 0.5, 2.0)) == (2.0, 2.0, 0.5)


def test_hidden_position_sits_under_the_floor_and_tolerates_nan() -> None:
    assert hidden_position((1.0, 2.0, 3.0)) == (1.0, 2.0, HIDDEN_Z)
    assert hidden_position((float("nan"), 2.0, 3.0)) == (0.0, 2.0, HIDDEN_Z)


def test_frame_mapping_between_native_frames_and_kubric() -> None:
    assert sim_frame_end(12, 24) == 35
    rows = list(range(36))[native_entries(12, 24)]
    assert rows[0] == 12 and rows[-1] == 35 and len(rows) == 24
    assert kubric_frame(0) == 1


def test_probe_points_surround_the_target() -> None:
    probes = probe_points((1.0, 2.0, 0.5))
    assert probes.shape == (16, 3)
    np.testing.assert_allclose(probes.mean(axis=0)[:2], (1.0, 2.0))


def test_state_table_keeps_the_native_frames_after_settling() -> None:
    settle = 3
    spec, end = _settled_scene(settle)
    table = state_table(spec, [_track(end + 1, 0.5)])  # type: ignore[arg-type]
    assert table.frame is CoordFrame.WORLD_ZUP
    assert table.position.shape == (24, 1, 3)
    assert table.position[0, 0, 2] == settle
    assert np.all(table.position_sigma == 0.0)
    np.testing.assert_allclose(table.size[5, 0], (0.4, 0.4, 0.4))
    np.testing.assert_allclose(table.gravity_dir, (0.0, 0.0, -1.0))
    assert np.isnan(table.acceleration).all()


def test_raw_collisions_keeps_one_row_per_event() -> None:
    events = [
        {"a": "x", "b": "y", "frame": 1.5, "position": (0, 0, 0), "normal": (0, 0, 1), "force": 2},
        {
            "a": "x",
            "b": "floor",
            "frame": 3.0,
            "position": (1, 0, 0),
            "normal": (0, 0, 1),
            "force": 0,
        },
    ]
    raw = raw_collisions("s", events)
    assert list(raw.a) == ["x", "x"] and list(raw.b) == ["y", "floor"]
    assert raw.position.shape == (2, 3) and raw.frame[1] == 3.0
    assert raw_collisions("s", []).position.shape == (0, 3)


def _states(visible: list[bool], positions: np.ndarray) -> ObjectStateTable:
    table = records.states()
    table = dataclasses.replace(
        table,
        position=positions,
        quaternion=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 2, 1)),
        visible=np.array([visible, [True, True], [True, True]]),
    )
    return table


def test_pose_keyframes_hide_unseen_objects_and_fill_unknown_values() -> None:
    positions = np.ones((3, 2, 3))
    positions[2, 1] = np.nan
    table = _states([False, True], positions)
    quaternion = table.quaternion.copy()
    quaternion[1, 0] = np.nan
    size = np.full((3, 2, 3), 0.2)
    size[0, 0] = np.nan
    table = dataclasses.replace(table, quaternion=quaternion, size=size)
    keys = pose_keyframes(table, 0, PRIMITIVE_BOUNDS, (0.4, 0.4, 0.4))
    assert keys[0][0][2] == HIDDEN_Z
    assert keys[0][2] == pytest.approx((0.2, 0.2, 0.2))
    assert keys[1][1] == IDENTITY_WXYZ
    assert keys[1][2] == pytest.approx((0.1, 0.1, 0.1))
    assert pose_keyframes(table, 1, PRIMITIVE_BOUNDS, (0.4, 0.4, 0.4))[2][0][2] == HIDDEN_Z


def test_present_objects_and_pairs_include_the_floor() -> None:
    positions = np.ones((3, 2, 3))
    table = _states([False, True], positions)
    assert present_objects(table, 0) == [1]
    assert present_objects(table, 1) == [0, 1]
    assert object_pairs([0, 1, 2]) == [(0, 1), (0, 2), (1, 2), (0, None), (1, None), (2, None)]


def test_xyzw_reorders_the_quaternion() -> None:
    assert xyzw((1.0, 2.0, 3.0, 4.0)) == [2.0, 3.0, 4.0, 1.0]


def test_closest_arrays_columns() -> None:
    rows = [(0, "a", "floor", (0, 0, 0), (0, 0, -0.001), (0, 0, 1), -0.001)]
    arrays = closest_arrays(rows)
    assert arrays["point_b"].shape == (1, 3) and arrays["distance"][0] == -0.001
    assert closest_arrays([])["point_a"].shape == (0, 3)


def test_to_uint8_rescales_sixteen_bit_images() -> None:
    assert to_uint8(np.array([65535, 0], dtype=np.uint16)).tolist() == [255, 0]
    assert to_uint8(np.array([200], dtype=np.uint8)).tolist() == [200]


def test_clip_inputs_rejects_mismatched_objects(tmp_path: Path) -> None:
    spec = records.scene()
    states = dataclasses.replace(records.states(), frame=CoordFrame.WORLD_ZUP)
    save_record(spec, tmp_path / "spec.json")
    save_record(states, tmp_path / "states.npz")
    job = {"spec_path": str(tmp_path / "spec.json"), "states_path": str(tmp_path / "states.npz")}
    with pytest.raises(ValueError, match="differ"):
        clip_inputs(job)


def test_guarded_reports_the_failure_and_goes_on(tmp_path: Path) -> None:
    job = {"failed_path": str(tmp_path / "failed.json")}

    def boom(_: dict[str, object]) -> None:
        raise RuntimeError("no GPU")

    assert guarded(job, boom) is False
    assert "RuntimeError: no GPU" in json.loads((tmp_path / "failed.json").read_text())["error"]
    assert guarded({"failed_path": str(tmp_path / "ok.json")}, lambda _: None) is True
    assert not (tmp_path / "ok.json").exists()
