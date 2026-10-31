"""The scene plane of ``contacts/kinematic_scene``: never under the body it judges, no
simulator constant, and its sigma."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import scene_contact_cases as cases

from physverify.extraction.providers._scene_plane import HeightError, ScenePlane, up_of
from physverify.physics.geometry import Plane
from physverify.physics.tracks import support_plane
from physverify.schema.contacts import ContactKind, ContactTable
from physverify.schema.states import ObjectStateTable

sc = cases.sc
TOL = 1e-9


def table(case: sc.Case, records: tuple) -> ContactTable:  # type: ignore[type-arg]
    states = case.states
    return ContactTable(
        clip_id=states.clip_id,
        frame=states.frame,
        unit=states.unit,
        source_key=states.source_key,
        records=records,
    )


# 5. Never the judged body's own plane


def test_one_object_alone_has_no_environment_record() -> None:
    case = cases.boxes_at([0.0])
    found = cases.outcome(case)
    assert found.planes == (None,)
    assert cases.env_records(found.records) == []


def test_two_objects_are_each_judged_against_the_other() -> None:
    lift = 0.02 * cases.LENGTH
    case = cases.boxes_at([0.0, lift])
    records = cases.outcome(case).records
    (low,) = cases.env_records(records, "box0")
    (high,) = cases.env_records(records, "box1")
    assert low.kind is high.kind is ContactKind.RESTING
    assert low.point_distances == pytest.approx(np.full(4, -lift), abs=TOL)
    assert high.point_distances == pytest.approx(np.full(4, lift), abs=TOL)
    assert cases.pair_records(records) == []


def test_a_sunk_body_sees_the_plane_of_the_others() -> None:
    depth = 0.3 * cases.LENGTH
    case = cases.boxes_at([0.0, 0.0, -depth, 0.0])
    contacts = table(case, cases.outcome(case).records)
    up, offset = cases.world_plane(case)
    plane = support_plane("box2", contacts)
    assert isinstance(plane, Plane)
    np.testing.assert_allclose(plane.normal, up, atol=TOL)
    assert plane.offset == pytest.approx(offset, abs=TOL)
    (record,) = cases.env_records(contacts.records, "box2")
    assert record.point_distances == pytest.approx(np.full(4, -depth), abs=TOL)
    assert record.max_penetration == pytest.approx(depth, abs=TOL)


def test_two_stacked_boxes_alone_rest_on_each_other_and_build_no_plane_below() -> None:
    case = cases.stacked_boxes()
    found = cases.outcome(case)
    (record,) = found.records
    assert (record.object_a, record.object_b, record.kind) == ("box0", "box1", ContactKind.RESTING)
    assert found.planes[0] is None


# 6. No simulator constant


def tilted(states: ObjectStateTable, rotation: np.ndarray, shift: np.ndarray) -> ObjectStateTable:
    """States moved rigidly by x -> R x + shift; orientations dropped (cubes are assumed
    upright from gravity)."""
    return replace(
        states,
        position=states.position @ rotation.T + shift,
        velocity=states.velocity @ rotation.T,
        acceleration=states.acceleration @ rotation.T,
        quaternion=np.full(states.quaternion.shape, np.nan),
        gravity_dir=rotation @ states.gravity_dir,
    )


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    axis = np.cross(a, b)
    s, c = float(np.linalg.norm(axis)), float(a @ b)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + k + k @ k * ((1.0 - c) / s**2)


@pytest.mark.parametrize("height", [0.37, 0.0, -1.3])
def test_the_plane_follows_gravity_and_the_floor_not_the_table_axes(height: float) -> None:
    angle = math.radians(20.0)
    target = np.array([0.0, math.sin(angle), math.cos(angle)])
    case = cases.boxes_at([0.0, 0.0])
    up, offset = cases.world_plane(case)
    rotation = rotation_between(up, target)
    states = tilted(case.states, rotation, (height - offset) * target)
    contacts = table(case, cases.outcome(case, states=states).records)
    for object_id in ("box0", "box1"):
        plane = support_plane(object_id, contacts)
        assert isinstance(plane, Plane)
        np.testing.assert_allclose(plane.normal, target, atol=TOL)
        assert plane.offset == pytest.approx(height, abs=TOL)


def test_the_size_of_gravity_does_not_matter() -> None:
    g = np.array([0.1, -0.9, 0.3])
    np.testing.assert_allclose(up_of(3.7 * g), up_of(g), atol=1e-15)
    assert up_of(np.full(3, np.nan)) is None


# 7. Plane sigma


def test_the_plane_sigma_adds_the_height_and_the_tilt_errors() -> None:
    up = np.array([0.0, 0.0, 1.0])
    centres = np.array([[0.0, 0.0, 0.1], [3.0, 4.0, 0.2]])
    plane = ScenePlane.of(up, {1: [0.01, 0.02, 0.03]}, centres, np.array([0.4, 0.05]), 0.02)
    support = plane.for_object(0)
    assert support is not None and support.source == 1
    assert support.height == pytest.approx(0.02)
    assert support.sigma == pytest.approx(math.hypot(0.05, 5.0 * 0.02))
    assert plane.for_object(1) is None
    error = HeightError(0.06, 0.08)
    assert error.sigma(2.0) == pytest.approx(2.0 * math.hypot(0.06, 0.04))


def test_an_unknown_tilt_error_leaves_records_with_unknown_confidence() -> None:
    case = cases.boxes_at([0.0, 0.0])
    states = replace(case.states, gravity_dir_sigma=math.nan)
    records = cases.env_records(cases.outcome(case, states=states).records)
    assert len(records) == 2
    assert all(math.isnan(r.confidence) for r in records)


def test_without_gravity_pairs_remain_and_there_is_no_environment() -> None:
    case = cases.stacked_boxes()
    floor = sc.build("floor", [cases.box("box2", 3 * cases.SPACING, 0.0)], [], 25.0).states
    states = cases.without_gravity(cases.cs.joined(case.states, floor))
    found = cases.outcome(case, states=states)
    assert cases.env_records(found.records) == []
    assert [(r.object_a, r.object_b) for r in found.records] == [("box0", "box1")]
