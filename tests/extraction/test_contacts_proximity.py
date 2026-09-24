"""AC10: contacts from the proximity of primitives fitted to the kinematic states."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from support import make_context, run_provider, sampling_of
from synthetic_clip import SyntheticClip

from physverify.extraction.matching import align_ground_truth
from physverify.extraction.providers.contacts_proximity import (
    Params,
    ProximityContacts,
    contact_probability,
    nan_median,
)
from physverify.physics.geometry import characteristic_lengths
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactTable, ManifoldKind
from physverify.schema.stages import Stage
from physverify.schema.states import CoordFrame, ObjectStateTable

BALL_RADIUS = 0.15


def truth_states(clip: SyntheticClip) -> ObjectStateTable:
    sampling = sampling_of(clip.manifest)
    truth = clip.ground_truth
    return align_ground_truth(truth, truth.masks(sampling), sampling).states


def contacts(clip: SyntheticClip, states: ObjectStateTable) -> ContactTable:
    output = run_provider(
        ProximityContacts, make_context(clip, upstream={Stage.KINEMATICS: states})
    )
    assert isinstance(output, ContactTable)
    return output


def sorted_rows(points: np.ndarray) -> np.ndarray:
    return points[np.lexsort(points.T[::-1])]


def test_truth_states_give_only_the_resting_box_on_the_ground(fixture_clip: SyntheticClip) -> None:
    table = contacts(fixture_clip, truth_states(fixture_clip))
    truth = fixture_clip.ground_truth.contacts(CoordFrame.CAMERA_OPENCV).records[0]
    assert [(r.object_a, r.object_b) for r in table.records] == [("box", ENV_ID)]
    record = table.records[0]
    assert record.kind is ContactKind.RESTING
    assert record.manifold is ManifoldKind.PRIMITIVE
    assert not record.orientation_assumed
    assert (record.t_start, record.t_end) == (0.0, 11.0)
    assert record.points.shape == truth.points.shape == (4, 3)
    np.testing.assert_allclose(
        sorted_rows(record.points), sorted_rows(truth.points), rtol=0, atol=1e-6
    )
    assert record.confidence == 1.0


def test_the_falling_ball_stops_just_outside_the_tolerance(fixture_clip: SyntheticClip) -> None:
    """The ball ends 0.0196 m above the ground, beyond tol = 0.05 L = 0.015 m: no ball contact."""
    world = fixture_clip.ground_truth.states(CoordFrame.WORLD_ZUP)
    ball = world.object_index("ball")
    gap = float(world.position[-1, ball, 2]) - BALL_RADIUS
    tol = Params().tol_rel * float(characteristic_lengths(world.size[-1, ball]))
    assert gap == pytest.approx(0.0196, abs=5e-4)
    assert tol == pytest.approx(0.015)
    assert gap > tol


def test_unknown_orientation_with_gravity_assumes_an_upright_pose(
    fixture_clip: SyntheticClip,
) -> None:
    states = truth_states(fixture_clip)
    unknown = replace(states, quaternion=np.full_like(states.quaternion, np.nan))
    records = contacts(fixture_clip, unknown).records
    assert [(r.object_a, r.object_b) for r in records] == [("box", ENV_ID)]
    assert records[0].orientation_assumed
    assert records[0].manifold is ManifoldKind.PRIMITIVE


def test_unknown_gravity_reduces_bodies_to_spheres_and_drops_the_ground(
    fixture_clip: SyntheticClip,
) -> None:
    states = truth_states(fixture_clip)
    ball, box = states.object_index("ball"), states.object_index("box")
    box_radius = characteristic_lengths(states.size[:, box]) / 2.0
    position = states.position.copy()
    position[:, ball] = states.position[:, box]
    position[:, ball, 0] += BALL_RADIUS + box_radius
    touching = replace(
        states,
        position=position,
        velocity=np.zeros_like(states.velocity),
        quaternion=np.full_like(states.quaternion, np.nan),
        gravity_dir=np.full(3, np.nan),
        gravity_dir_sigma=math.nan,
    )
    records = contacts(fixture_clip, touching).records
    assert [(r.object_a, r.object_b) for r in records] == [("ball", "box")]
    assert records[0].manifold is ManifoldKind.POINT
    assert records[0].points.shape == (1, 3)
    assert records[0].orientation_assumed


def test_contact_probability_limits() -> None:
    assert contact_probability(0.01, 0.015, 0.0) == 1.0
    assert contact_probability(0.02, 0.015, 0.0) == 0.0
    assert contact_probability(0.015, 0.015, 0.003) == pytest.approx(0.5)
    assert math.isnan(contact_probability(0.0, 0.015, math.nan))


def test_nan_median_skips_nan_and_is_nan_when_all_are_nan() -> None:
    assert nan_median([1.0, math.nan, 3.0]) == 2.0
    assert math.isnan(nan_median([math.nan, math.nan]))
