"""Closest points between every pair of bodies (floor included) on the poses of an edited
trajectory, with PyBullet's ``getClosestPoints``.

Job keys: clip_id, spec_path, states_path, out_path, failed_path, tol, floor. Each body keeps
the collision shape of its ``SceneSpec`` size (PyBullet shapes cannot be rescaled); objects that
are not visible in a frame are skipped. ``closest.npz`` is the job's marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from physverify.kubric_worker.build_scene import (
    body_ids,
    clip_inputs,
    guarded,
    open_scene,
    save_arrays,
)
from physverify.schema.constants import ENV_ID

if TYPE_CHECKING:
    from collections.abc import Sequence

    from physverify.schema.states import ObjectStateTable

POS_ON_A, POS_ON_B, NORMAL_ON_B, DISTANCE = 5, 6, 7, 8


def run(job: dict[str, Any]) -> None:
    guarded(job, closest_points)


def closest_points(job: dict[str, Any]) -> None:
    import pybullet as pb

    spec, states = clip_inputs(job)
    ids = [o.object_id for o in spec.objects]
    with open_scene(spec, job["floor"], simulate=True) as built:
        client = built.simulator.physics_client
        bodies, floor = body_ids(built)
        rows: list[tuple[Any, ...]] = []
        for t in range(states.t.shape[0]):
            present = present_objects(states, t)
            for k in present:
                pb.resetBasePositionAndOrientation(
                    bodies[k],
                    states.position[t, k].tolist(),
                    xyzw(states.quaternion[t, k]),
                    physicsClientId=client,
                )
            for a, b in object_pairs(present):
                other = floor if b is None else bodies[b]
                name_b = ENV_ID if b is None else ids[b]
                for point in pb.getClosestPoints(
                    bodies[a], other, float(job["tol"]), physicsClientId=client
                ):
                    rows.append((t, ids[a], name_b, *_point(point)))
    save_arrays(Path(job["out_path"]), closest_arrays(rows))


def present_objects(states: ObjectStateTable, t: int) -> list[int]:
    """Objects drawn in frame ``t``: visible with a finite pose."""
    finite = np.isfinite(states.position[t]).all(axis=1) & np.isfinite(states.quaternion[t]).all(
        axis=1
    )
    return [int(k) for k in np.flatnonzero(finite & states.visible[t])]


def object_pairs(present: Sequence[int]) -> list[tuple[int, int | None]]:
    """Every pair (i < j) of present objects plus each present object against the floor (None)."""
    pairs: list[tuple[int, int | None]] = [
        (i, j) for n, i in enumerate(present) for j in present[n + 1 :]
    ]
    return pairs + [(i, None) for i in present]


def xyzw(quaternion_wxyz: Sequence[float]) -> list[float]:
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    return [x, y, z, w]


def _point(point: Sequence[Any]) -> tuple[Any, ...]:
    return (point[POS_ON_A], point[POS_ON_B], point[NORMAL_ON_B], point[DISTANCE])


def closest_arrays(rows: Sequence[Sequence[Any]]) -> dict[str, np.ndarray]:
    """Columns read by ``synth.contacts_gt.ClosestSamples``; the normal on b points from b to a."""

    def vectors(column: int) -> np.ndarray:
        return np.asarray([r[column] for r in rows], dtype=np.float64).reshape(len(rows), 3)

    return {
        "frame": np.asarray([r[0] for r in rows], dtype=np.int64).reshape(-1),
        "a": np.asarray([str(r[1]) for r in rows], dtype=str).reshape(-1),
        "b": np.asarray([str(r[2]) for r in rows], dtype=str).reshape(-1),
        "point_a": vectors(3),
        "point_b": vectors(4),
        "normal_on_b": vectors(5),
        "distance": np.asarray([r[6] for r in rows], dtype=np.float64).reshape(-1),
    }
