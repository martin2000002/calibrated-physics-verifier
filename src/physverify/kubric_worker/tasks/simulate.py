"""Simulate one scene per job with PyBullet (through Kubric) and write the world-frame states.

Job keys: sim_id, spec_path, states_path, raw_path, meta_path, failed_path, floor.
``sim_meta.json`` is written last and marks the job done.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from physverify.kubric_worker.build_scene import (
    guarded,
    native_entries,
    open_scene,
    save_record,
    sim_frame_end,
    write_json,
)
from physverify.schema.contacts import RawCollisions
from physverify.schema.scene import SceneSpec
from physverify.schema.states import (
    CoordFrame,
    LengthUnit,
    ObjectStateTable,
    ground_truth_source_key,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

TRACK_KEYS = ("position", "quaternion", "velocity", "angular_velocity")


def run(job: dict[str, Any]) -> None:
    guarded(job, simulate)


def simulate(job: dict[str, Any]) -> None:
    spec = SceneSpec.load_json(job["spec_path"])
    with open_scene(spec, job["floor"], simulate=True) as built:
        start = time.perf_counter()
        animation, events = built.simulator.run(
            frame_start=0, frame_end=sim_frame_end(spec.settle_frames, spec.n_frames)
        )
        seconds = time.perf_counter() - start
        tracks = [animation[asset] for asset in built.objects]
        rows = [(event["instances"], event) for event in events]
        raw = raw_collisions(
            spec.scene_id,
            [
                _named(instances, event)
                for instances, event in rows
                if instances[0] is not None and instances[1] is not None
            ],
        )
    save_record(state_table(spec, tracks), Path(job["states_path"]))
    save_record(raw, Path(job["raw_path"]))
    write_json(
        Path(job["meta_path"]),
        {
            "sim_id": job["sim_id"],
            "seconds": seconds,
            "kubric_frames": [0, sim_frame_end(spec.settle_frames, spec.n_frames)],
            "settle_frames": spec.settle_frames,
            "steps_per_frame": spec.step_rate // spec.frame_rate,
            "n_events": int(raw.frame.shape[0]),
            "damping": 0.0,
        },
    )


def _named(instances: Sequence[Any], event: Mapping[str, Any]) -> dict[str, Any]:
    """Kubric lists ``(asset_b, asset_a)`` with the point on b and the normal on b (b to a)."""
    return {
        "a": instances[1].name,
        "b": instances[0].name,
        "frame": event["frame"],
        "position": event["position"],
        "normal": event["contact_normal"],
        "force": event["force"],
    }


def raw_collisions(scene_id: str, events: Sequence[Mapping[str, Any]]) -> RawCollisions:
    def column(key: str, width: int = 0) -> np.ndarray:
        shape = (len(events), width) if width else (len(events),)
        return np.asarray([e[key] for e in events], dtype=np.float64).reshape(shape)

    return RawCollisions(
        scene_id=scene_id,
        a=np.asarray([str(e["a"]) for e in events], dtype=str).reshape(-1),
        b=np.asarray([str(e["b"]) for e in events], dtype=str).reshape(-1),
        frame=column("frame"),
        position=column("position", 3),
        normal=column("normal", 3),
        force=column("force"),
    )


def state_table(
    spec: SceneSpec, tracks: Sequence[Mapping[str, Sequence[Sequence[float]]]]
) -> ObjectStateTable:
    """Native frames of the simulated tracks as a WORLD table with exact (zero-sigma) values;
    acceleration is left to the host (central differences at export)."""
    rows = native_entries(spec.settle_frames, spec.n_frames)
    n_t, n_obj = spec.n_frames, len(spec.objects)
    arrays: dict[str, Any] = {
        key: np.stack([np.asarray(track[key], dtype=np.float64)[rows] for track in tracks], axis=1)
        for key in TRACK_KEYS
    }
    gravity = np.asarray(spec.gravity, dtype=np.float64)
    blank = ObjectStateTable.blank(
        clip_id=spec.scene_id,
        frame=CoordFrame.WORLD_ZUP,
        unit=LengthUnit.METER,
        source_key=ground_truth_source_key(spec.scene_id),
        t=np.arange(n_t, dtype=np.float64) / spec.frame_rate,
        frame_index=np.arange(n_t, dtype=np.int64),
        object_ids=[o.object_id for o in spec.objects],
        labels=[o.label for o in spec.objects],
        primitive=[o.primitive.value for o in spec.objects],
    )
    zeros3 = np.zeros((n_t, n_obj, 3))
    return dataclasses.replace(
        blank,
        **arrays,
        position_sigma=zeros3,
        velocity_sigma=zeros3,
        size=np.broadcast_to(
            np.asarray([o.extents for o in spec.objects], dtype=np.float64), (n_t, n_obj, 3)
        ).copy(),
        size_sigma=zeros3,
        orientation_sigma=np.zeros((n_t, n_obj)),
        occlusion=np.zeros((n_t, n_obj)),
        confidence=np.ones((n_t, n_obj)),
        gravity_dir=gravity / np.linalg.norm(gravity),
        gravity_dir_sigma=0.0,
    )
