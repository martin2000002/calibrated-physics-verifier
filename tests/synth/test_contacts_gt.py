"""Contact ground truth from Kubric's raw events and from closest points."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from physverify.physics.contact_manifold import contact_manifold, convex_hull_indices
from physverify.physics.geometry import Plane, project_to_plane
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ManifoldKind, RawCollisions
from physverify.schema.scene import SceneSpec
from physverify.schema.states import ObjectStateTable
from physverify.synth.frame_windows import solid_at
from physverify.synth.gt_contact_records import CLOSEST_SOURCE
from physverify.synth.gt_contacts import (
    EVENT_STEP_OFFSET,
    ClosestSamples,
    Patch,
    classify_interval,
    contact_table,
    event_frames_to_native,
    from_closest,
    from_raw_events,
    split_intervals,
    split_runs,
)
from physverify.synth.settings import ContactSettings, DatasetSettings

DATA = Path(__file__).parent / "data"


def load_sim(family: str) -> tuple[SceneSpec, ObjectStateTable, RawCollisions]:
    directory = DATA / "clean" / family
    return (
        SceneSpec.load_json(directory / "spec.json"),
        ObjectStateTable.load_npz(directory / "states.npz"),
        RawCollisions.load_npz(directory / "raw_collisions.npz"),
    )


def floor_impact_frames(scene: SceneSpec, states: ObjectStateTable) -> list[float]:
    """Frames at which ball 0 reaches the floor, from the free flight before each bounce."""
    g = -scene.gravity[2]
    radius = scene.objects[0].extents[2] / 2.0
    z, vz = states.position[:, 0, 2], states.velocity[:, 0, 2]
    out = []
    for k in np.flatnonzero((vz[:-1] < 0.0) & (vz[1:] > 0.0)):
        drop = z[k] - radius
        tau = (vz[k] + math.sqrt(vz[k] ** 2 + 2.0 * g * drop)) / g
        out.append(float(k) + tau * scene.frame_rate)
    return out


def test_event_frames_are_shifted_to_the_middle_of_the_step() -> None:
    frames = np.array([10.0, 12.5])
    native = event_frames_to_native(frames, 10, steps_per_frame=10)
    assert native == pytest.approx([EVENT_STEP_OFFSET / 10, 2.5 + EVENT_STEP_OFFSET / 10])


def test_split_intervals_cuts_at_gaps() -> None:
    frames = np.array([0.0, 0.1, 0.2, 3.0, 3.1, 9.0])
    runs = split_intervals(frames, 0.5)
    assert [(s.start, s.stop) for s in runs] == [(0, 3), (3, 5), (5, 6)]
    assert split_intervals(np.array([]), 0.5) == []


def test_short_intervals_are_impacts(ds: DatasetSettings) -> None:
    cfg = ds.contacts
    assert classify_interval(0.0, cfg) is ContactKind.IMPACT
    assert classify_interval(cfg.impact_max_frames, cfg) is ContactKind.RESTING


def test_floor_impacts_land_on_the_closed_form_times(ds: DatasetSettings) -> None:
    scene, states, raw = load_sim("f2_bounce")
    table = from_raw_events(raw, scene, states, ds.contacts)
    impacts = [
        r
        for r in table.records
        if r.kind is ContactKind.IMPACT and {r.object_a, r.object_b} == {"ball_0", ENV_ID}
    ]
    expected = floor_impact_frames(scene, states)
    assert len(impacts) == 2
    for record, frame in zip(impacts, expected, strict=True):
        assert record.t_start == pytest.approx(frame, abs=0.1)
    resting = [r for r in table.records if r.kind is ContactKind.RESTING]
    assert [(r.object_a, r.object_b) for r in resting] == [("cube_0", ENV_ID)]


def test_ball_to_ball_collision_is_one_impact(ds: DatasetSettings) -> None:
    scene, states, raw = load_sim("f3_collision")
    table = from_raw_events(raw, scene, states, ds.contacts)
    between = [r for r in table.records if ENV_ID not in (r.object_a, r.object_b)]
    assert len(between) == 1
    assert between[0].kind is ContactKind.IMPACT
    frame = round(between[0].t_start)
    gap = np.linalg.norm(states.position[frame, 0] - states.position[frame, 1])
    radii = (scene.objects[0].extents[0] + scene.objects[1].extents[0]) / 2.0
    assert gap == pytest.approx(radii, rel=0.25)


def closest_rows(frames: list[int], distance: float) -> ClosestSamples:
    n = len(frames)
    return ClosestSamples(
        frame=np.asarray(frames, dtype=np.int64),
        a=np.asarray(["cube_0"] * n),
        b=np.asarray([ENV_ID] * n),
        point_a=np.zeros((n, 3)),
        point_b=np.zeros((n, 3)),
        normal_on_b=np.tile([0.0, 0.0, 1.0], (n, 1)),
        distance=np.full(n, distance),
    )


def test_closest_points_give_resting_contact_and_keep_impacts(ds: DatasetSettings) -> None:
    scene, states, raw = load_sim("f2_bounce")
    provisional = from_raw_events(raw, scene, states, ds.contacts)
    frames = list(range(states.t.shape[0]))
    table = from_closest(closest_rows(frames, 0.0), provisional, states, ds.contacts)
    resting = [r for r in table.records if r.kind is ContactKind.RESTING]
    assert len(resting) == 1
    assert (resting[0].t_start, resting[0].t_end) == (0.0, float(frames[-1]))
    impacts = [r for r in table.records if r.kind is ContactKind.IMPACT]
    assert impacts == [r for r in provisional.records if r.kind is ContactKind.IMPACT]
    far = from_closest(closest_rows(frames, 1.0), provisional, states, ds.contacts)
    assert all(r.kind is ContactKind.IMPACT for r in far.records)


CUBE = "cube_0"
UP = np.array([0.0, 0.0, 1.0])


def raw_events(scene: SceneSpec, rows: list[tuple[float, str, str, Any, Any, float]]) -> Any:
    """Kubric-style event rows from (native frame, a, b, point, normal, force)."""
    spf = scene.step_rate // scene.frame_rate
    native = np.array([r[0] for r in rows], dtype=np.float64)
    return RawCollisions(
        scene_id=scene.scene_id,
        a=np.array([r[1] for r in rows]),
        b=np.array([r[2] for r in rows]),
        frame=native + scene.settle_frames - EVENT_STEP_OFFSET / spf,
        position=np.array([r[3] for r in rows], dtype=np.float64).reshape(-1, 3),
        normal=np.array([r[4] for r in rows], dtype=np.float64).reshape(-1, 3),
        force=np.array([r[5] for r in rows], dtype=np.float64),
    )


def corners(states: ObjectStateTable) -> Any:
    """The four bottom corners of the resting cube, from its primitive against the floor."""
    manifold = contact_manifold(
        solid_at(states, states.object_index(CUBE), 0), Plane.ground(), 1e-3
    )
    assert manifold is not None and manifold.points.shape[0] == 4
    return manifold.points


def steps(scene: SceneSpec, frames: float) -> Any:
    return np.arange(int(frames * scene.step_rate / scene.frame_rate)) / (
        scene.step_rate / scene.frame_rate
    )


def cube_records(table: Any) -> list[ContactRecord]:
    return [r for r in table.records if (r.object_a, r.object_b) == (CUBE, ENV_ID)]


def resting_cube(
    bank: Any, cfg: ContactSettings, points_of_step: Any, force: Any = lambda k: 10.0
) -> list[ContactRecord]:
    clip = bank.analytic("f4_stack")
    rows = [
        (float(t), CUBE, ENV_ID, point, -UP, force(k))
        for k, t in enumerate(steps(clip.scene, 4.0))
        for point in points_of_step(k)
    ]
    return cube_records(from_raw_events(raw_events(clip.scene, rows), clip.scene, clip.states, cfg))


def test_the_recorded_cube_on_the_floor_is_one_resting_record(bank: Any) -> None:
    records = cube_records(bank.recorded("f4_stack").contacts)
    assert [r.kind for r in records] == [ContactKind.RESTING]


def test_alternating_point_counts_give_one_resting_record(bank: Any, clean_ds: Any) -> None:
    clip = bank.analytic("f4_stack")
    quad = corners(clip.states)
    subsets = ([0], [0, 1, 2], [0, 1, 2, 3])
    records = resting_cube(bank, clean_ds.contacts, lambda k: quad[subsets[k % 3]])
    assert [r.kind for r in records] == [ContactKind.RESTING]
    assert records[0].manifold is ManifoldKind.MEASURED
    assert records[0].points.shape == (4, 3)
    assert records[0].normal == pytest.approx(UP, abs=1e-9)


def test_event_records_keep_the_peak_force_and_at_most_eight_points(
    bank: Any, clean_ds: Any
) -> None:
    clip = bank.analytic("f4_stack")
    centre = corners(clip.states).mean(axis=0)
    ring = [
        centre + 0.04 * np.array([math.cos(a), math.sin(a), 0.0])
        for a in np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
    ]
    records = resting_cube(bank, clean_ds.contacts, lambda k: ring, force=lambda k: float(k % 7))
    assert len(records) == 1
    assert records[0].peak_force == 6.0
    assert 3 <= records[0].points.shape[0] <= 8


@pytest.mark.parametrize("order", ["floor-first", "cube-first"])
def test_normals_point_from_b_to_a(bank: Any, clean_ds: Any, order: str) -> None:
    clip = bank.analytic("f4_stack")
    quad = corners(clip.states)
    pair = (ENV_ID, CUBE) if order == "floor-first" else (CUBE, ENV_ID)
    rows = [(float(t), *pair, p, -UP, 5.0) for t in steps(clip.scene, 2.0) for p in quad]
    table = from_raw_events(
        raw_events(clip.scene, rows), clip.scene, clip.states, clean_ds.contacts
    )
    (record,) = table.records
    assert (record.object_a, record.object_b) == (CUBE, ENV_ID)
    assert record.normal == pytest.approx(UP, abs=1e-9)
    assert record.point_normals == pytest.approx(np.tile(UP, (4, 1)), abs=1e-9)


def tilted(degrees: float) -> Any:
    angle = math.radians(degrees)
    return np.array([math.sin(angle), 0.0, math.cos(angle)])


@pytest.mark.parametrize(
    ("change", "runs"),
    [
        ({"normal": tilted(15.0)}, 2),
        ({"normal": tilted(5.0)}, 1),
        ({"area": 0.014}, 2),
        ({"area": 0.011}, 1),
        ({"centroid": np.array([0.02, 0.0, 0.0])}, 2),
        ({"centroid": np.array([0.005, 0.0, 0.0])}, 1),
    ],
)
def test_split_runs_cut_on_normal_area_and_drift(
    clean_ds: Any, change: dict[str, Any], runs: int
) -> None:
    base = Patch(UP, 0.01, np.zeros(3))
    other = dataclasses.replace(base, **change)
    assert len(split_runs([base, base, other, other], clean_ds.contacts)) == runs


def test_split_runs_measure_drift_from_the_first_patch_of_the_run(clean_ds: Any) -> None:
    patches = [Patch(UP, 0.01, np.array([0.006 * k, 0.0, 0.0])) for k in range(4)]
    assert split_runs(patches, clean_ds.contacts) == [range(2), range(2, 4)]


def test_a_sparse_face_contact_falls_back_to_the_primitive_patch(bank: Any, clean_ds: Any) -> None:
    clip = bank.analytic("f4_stack")
    corner = corners(clip.states)[0]
    (record,) = resting_cube(bank, clean_ds.contacts, lambda k: [corner])
    assert record.manifold is ManifoldKind.PRIMITIVE
    assert record.source == "primitive:sparse_pybullet_events"
    assert record.points.shape == (4, 3)
    assert record.max_penetration == pytest.approx(0.0, abs=1e-9)


def test_events_while_settling_are_dropped(bank: Any, clean_ds: Any) -> None:
    clip = bank.analytic("f4_stack")
    quad = corners(clip.states)
    settle = [
        (float(t) - 3.0, CUBE, ENV_ID, p, -UP, 50.0) for t in steps(clip.scene, 2.0) for p in quad
    ]
    table = from_raw_events(
        raw_events(clip.scene, settle), clip.scene, clip.states, clean_ds.contacts
    )
    assert table.records == ()
    live = [(float(t), CUBE, ENV_ID, p, -UP, 5.0) for t in steps(clip.scene, 2.0) for p in quad]
    both = from_raw_events(
        raw_events(clip.scene, settle + live), clip.scene, clip.states, clean_ds.contacts
    )
    (record,) = both.records
    assert record.t_start == pytest.approx(0.0, abs=1e-9)
    assert record.peak_force == 5.0


def test_event_penetration_is_measured_on_the_primitives(bank: Any, clean_ds: Any) -> None:
    clip = bank.analytic("f4_stack")
    sink = np.zeros_like(clip.states.position)
    sink[:, clip.states.object_index(CUBE), 2] = -0.001
    sunk = dataclasses.replace(clip.states, position=clip.states.position + sink)
    quad = corners(clip.states)
    rows = [(float(t), CUBE, ENV_ID, p, -UP, 5.0) for t in steps(clip.scene, 2.0) for p in quad]
    table = from_raw_events(raw_events(clip.scene, rows), clip.scene, sunk, clean_ds.contacts)
    (record,) = cube_records(table)
    assert record.point_distances is None
    assert record.max_penetration == pytest.approx(0.001, abs=1e-9)


CYLINDER = "cylinder_1"


def pair_rows(
    object_id: str, frames: Sequence[int], points: list[Any], distances: list[float]
) -> ClosestSamples:
    """``getClosestPoints`` rows against the floor at ``frames``, cycling through ``points``."""
    picks = [k % len(points) for k in range(len(frames))]
    point = np.array([points[k] for k in picks], dtype=np.float64)
    n = len(frames)
    return ClosestSamples(
        frame=np.asarray(list(frames), dtype=np.int64),
        a=np.asarray([object_id] * n),
        b=np.asarray([ENV_ID] * n),
        point_a=point,
        point_b=point.copy(),
        normal_on_b=np.tile(UP, (n, 1)),
        distance=np.array([distances[k % len(distances)] for k in picks], dtype=np.float64),
    )


def closest_record(states: ObjectStateTable, samples: ClosestSamples, cfg: Any) -> ContactRecord:
    table = from_closest(samples, contact_table(states.clip_id, []), states, cfg)
    (record,) = [r for r in table.records if r.kind is ContactKind.RESTING]
    return record


def inside(point: Any, polygon: Any) -> bool:
    hull = polygon[convex_hull_indices(polygon)]
    edges = np.roll(hull, -1, axis=0) - hull
    to_point = point - hull
    return bool(np.all(edges[:, 0] * to_point[:, 1] - edges[:, 1] * to_point[:, 0] > 0.0))


def rim(states: ObjectStateTable, object_id: str, degrees: list[float]) -> list[Any]:
    index = states.object_index(object_id)
    centre, radius = states.position[0, index], float(states.size[0, index, 0]) / 2.0
    return [
        np.array([centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a), 0.0])
        for a in np.radians(degrees)
    ]


def test_an_upright_cylinder_from_closest_points_takes_the_primitive_ring(
    bank: Any, clean_ds: Any
) -> None:
    states = bank.analytic("f4_stack").states
    points = rim(states, CYLINDER, [0.0, 20.0, 40.0])
    distances = [0.0, -0.0004, 0.0]
    samples = pair_rows(CYLINDER, range(states.n_frames), points, distances)
    record = closest_record(states, samples, clean_ds.contacts)
    assert record.manifold is ManifoldKind.PRIMITIVE
    assert record.source == f"primitive:{CLOSEST_SOURCE}"
    assert record.points.shape[0] >= 6
    centre = states.position[0, states.object_index(CYLINDER)]
    polygon = project_to_plane(record.points, UP)
    assert inside(project_to_plane(centre, UP)[0], polygon)
    assert not inside(project_to_plane(centre, UP)[0], project_to_plane(np.array(points), UP))
    assert record.max_penetration == pytest.approx(max(0.0, -min(distances)), abs=1e-12)


def test_a_resting_sphere_keeps_the_closest_points(bank: Any, clean_ds: Any) -> None:
    states = bank.analytic("f2_bounce").states
    index = states.object_index("ball_0")
    contact = states.position[0, index] - [0.0, 0.0, float(states.size[0, index, 2]) / 2.0]
    samples = pair_rows("ball_0", range(4, 10), [contact], [0.0])
    record = closest_record(states, samples, clean_ds.contacts)
    assert record.manifold is ManifoldKind.MEASURED
    assert record.source == CLOSEST_SOURCE
    assert record.points == pytest.approx(contact.reshape(1, 3))


def test_a_tilted_cylinder_keeps_the_closest_points(bank: Any, clean_ds: Any) -> None:
    clip = bank.analytic("f4_stack")
    half = math.radians(15.0)
    quaternion = clip.states.quaternion.copy()
    quaternion[:, clip.states.object_index(CYLINDER)] = [math.cos(half), math.sin(half), 0.0, 0.0]
    states = dataclasses.replace(clip.states, quaternion=quaternion)
    points = rim(states, CYLINDER, [0.0, 20.0, 40.0])
    samples = pair_rows(CYLINDER, range(states.n_frames), points, [0.0])
    record = closest_record(states, samples, clean_ds.contacts)
    assert record.manifold is ManifoldKind.MEASURED
    assert record.source == CLOSEST_SOURCE
    assert sorted(map(tuple, record.points)) == pytest.approx(sorted(map(tuple, points)))


def test_a_face_contact_without_a_primitive_manifold_keeps_the_closest_points(
    bank: Any, clean_ds: Any
) -> None:
    """Bullet reports the cube within tol_m (its collision margin) while the primitives stand
    1 cm apart: there is no primitive patch to take, so Bullet's points stay."""
    clip = bank.analytic("f4_stack")
    lift = np.zeros_like(clip.states.position)
    lift[:, clip.states.object_index(CUBE), 2] = 0.01
    states = dataclasses.replace(clip.states, position=clip.states.position + lift)
    quad = list(corners(clip.states))[:3]
    frames = [f for f in range(states.n_frames) for _ in quad]
    samples = pair_rows(CUBE, frames, quad, [0.001])
    assert clean_ds.contacts.tol_m >= 0.001
    record = closest_record(states, samples, clean_ds.contacts)
    assert record.manifold is ManifoldKind.MEASURED
    assert record.source == CLOSEST_SOURCE
    assert record.points.shape == (3, 3)
