"""WP-I: a ball-ball collision keeps a row's velocity only when a floor record overlaps the
interval between the row and t_c or the body is within the contact tolerance of the floor there;
every other row is extrapolated ballistically to t_c."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from reference.legacy_contact_rules import floor_reach, legacy_collision_velocities

from physverify.physics.measures.collision_energy import kinetic_energy_gain
from physverify.schema.constants import ENV_ID
from physverify.schema.contacts import ContactKind, ContactRecord, ContactTable
from physverify.schema.states import ObjectStateTable, PrimitiveKind
from physverify.synth.perturb import instantiate
from physverify.synth.perturbations._ballistics import impact_velocity
from physverify.synth.perturbations._measure import (
    collision_row,
    collision_velocities,
    floor_gap,
    pair_energy_gain,
    row_on_floor,
    row_overlaps_floor,
)
from physverify.synth.perturbations._state_edit import edit_object, gravity_of, radius_of, spheres

TOL = 1e-12
CONTACT_TOL = 0.002
K = 10
PHASE = 0.4
FLIGHT_GAP = 0.18
RATIO = 1.3
Vec = np.ndarray


@dataclasses.dataclass(frozen=True)
class Pair:
    """Two balls colliding at native frame K + PHASE whose rows K and K + 1 lie on the exact
    ballistic arcs through (t_c, v-) and (t_c, v+)."""

    states: ObjectStateTable
    impact: ContactRecord
    a: int
    b: int
    g: Vec
    t_c: float
    v_minus: tuple[Vec, Vec]
    v_plus: tuple[Vec, Vec]

    def contacts(self, *floor: tuple[int, float, float, ContactKind]) -> ContactTable:
        records = [self.impact] + [
            dataclasses.replace(
                self.impact,
                object_a=str(self.states.object_ids[index]),
                object_b=ENV_ID,
                t_start=start,
                t_end=end,
                kind=kind,
            )
            for index, start, end, kind in floor
        ]
        return ContactTable(
            clip_id=self.states.clip_id,
            frame=self.states.frame,
            unit=self.states.unit,
            source_key=self.states.source_key,
            records=tuple(records),
        )

    def at_height(self, index: int, gaps: tuple[float, float]) -> Pair:
        z = np.array([gap + radius_of(self.states, index) for gap in gaps])
        states = edit_object(self.states, index, slice(K, K + 2), position=self._xy(index, z))
        return dataclasses.replace(self, states=states)

    def _xy(self, index: int, z: Vec) -> Vec:
        rows = np.array(self.states.position[K : K + 2, index], copy=True)
        rows[:, 2] = z
        return rows

    def nominal(self) -> float:
        (am, bm), (ap, bp) = self.v_minus, self.v_plus
        return kinetic_energy_gain(RATIO, am, bm, ap, bp)

    def gain(self, contacts: ContactTable) -> float:
        return pair_energy_gain(
            self.states, contacts, (self.a, self.b), RATIO, self.t_c, self.g, CONTACT_TOL
        )


@pytest.fixture
def pair(bank: Any) -> Pair:
    clip = bank.analytic("f3_collision")
    states, g = clip.states, gravity_of(clip.scene)
    a, b = spheres(states)[:2]
    fps = float(clip.scene.frame_rate)
    t_c = (K + PHASE) / fps
    v_minus = (np.array([1.5, 0.0, -0.4]), np.array([-0.5, 0.2, 0.1]))
    v_plus = (np.array([-0.3, 0.1, 1.2]), np.array([0.9, -0.3, 0.6]))
    rows = np.array([K, K + 1])
    for index, before, after in ((a, v_minus[0], v_plus[0]), (b, v_minus[1], v_plus[1])):
        velocity = np.stack([before + g * (states.t[K] - t_c), after + g * (states.t[K + 1] - t_c)])
        z = np.full(2, 1.0 + radius_of(states, index))
        position = np.array(states.position[rows, index], copy=True)
        position[:, 2] = z
        states = edit_object(states, index, rows, velocity=velocity, position=position)
    impact = next(r for r in clip.contacts.records if r.kind is ContactKind.IMPACT)
    impact = dataclasses.replace(
        impact,
        object_a=str(states.object_ids[a]),
        object_b=str(states.object_ids[b]),
        t_start=K + PHASE,
        t_end=K + PHASE,
    )
    return Pair(states, impact, a, b, g, t_c, v_minus, v_plus)


def velocities(pair: Pair, contacts: ContactTable, index: int) -> tuple[Vec, Vec]:
    return collision_velocities(pair.states, contacts, index, pair.t_c, pair.g, CONTACT_TOL)


def test_the_collision_row_brackets_t_c(pair: Pair) -> None:
    assert collision_row(pair.states, pair.t_c) == K


def test_a_ball_that_bounces_two_rows_later_is_extrapolated_to_the_nominal(pair: Pair) -> None:
    """C5 a: the f3 pattern of s005/s025/s027, a floor bounce at K + 1.6 within the old margin."""
    pair = pair.at_height(pair.a, (0.5, FLIGHT_GAP))
    contacts = pair.contacts((pair.a, K + 1.6, K + 1.6, ContactKind.IMPACT))
    v_minus, v_plus = velocities(pair, contacts, pair.a)
    assert np.allclose(v_minus, pair.v_minus[0], atol=TOL, rtol=0.0)
    assert np.allclose(v_plus, pair.v_plus[0], atol=TOL, rtol=0.0)
    assert abs(pair.gain(contacts) - pair.nominal()) <= TOL
    _, old_plus = legacy_collision_velocities(pair.states, contacts, pair.a, pair.t_c, pair.g)
    bias = pair.g * (float(pair.states.t[K + 1]) - pair.t_c)
    assert np.allclose(old_plus - v_plus, bias, atol=TOL, rtol=0.0)


def test_a_struck_resting_body_keeps_its_row_values(pair: Pair) -> None:
    """C5 b: a RESTING record across the impact holds both rows."""
    contacts = pair.contacts((pair.b, 0.0, K + 3.0, ContactKind.RESTING))
    v_minus, v_plus = velocities(pair, contacts, pair.b)
    assert np.array_equal(v_minus, pair.states.velocity[K, pair.b])
    assert np.array_equal(v_plus, pair.states.velocity[K + 1, pair.b])


def test_without_a_floor_contact_nearby_the_result_is_the_old_one(pair: Pair) -> None:
    """C5 c: the old and new rules agree bit for bit away from the floor."""
    contacts = pair.contacts((pair.a, K + 8.0, K + 9.0, ContactKind.IMPACT))
    for index in (pair.a, pair.b):
        new = velocities(pair, contacts, index)
        old = legacy_collision_velocities(pair.states, contacts, index, pair.t_c, pair.g)
        assert all(np.array_equal(x, y) for x, y in zip(new, old, strict=True))
        expected = [
            impact_velocity(
                pair.states.velocity[row, index], float(pair.states.t[row]), pair.t_c, pair.g
            )
            for row in (K, K + 1)
        ]
        assert all(np.array_equal(x, y) for x, y in zip(new, expected, strict=True))


def test_a_record_ending_on_the_row_holds_only_that_row(pair: Pair) -> None:
    """C5 d: the interval of row K is [K, t_c], so a record ending at K touches it, not K + 1."""
    contacts = pair.contacts((pair.a, K - 4.0, float(K), ContactKind.RESTING))
    assert row_overlaps_floor(pair.states, contacts, pair.a, K, pair.t_c)
    assert not row_overlaps_floor(pair.states, contacts, pair.a, K + 1, pair.t_c)
    v_minus, v_plus = velocities(pair, contacts, pair.a)
    assert np.array_equal(v_minus, pair.states.velocity[K, pair.a])
    assert np.allclose(v_plus, pair.v_plus[0], atol=TOL, rtol=0.0)


@pytest.mark.parametrize("offset", [-1e-12, 1e-12])
def test_a_record_at_t_c_holds_both_rows(pair: Pair, offset: float) -> None:
    """C5 d: a record starting or ending within 1e-12 frames of t_c overlaps both intervals."""
    edge = K + PHASE + offset
    for record in ((pair.a, edge, K + 5.0), (pair.a, K - 5.0, edge)):
        contacts = pair.contacts((*record, ContactKind.RESTING))
        for row in (K, K + 1):
            assert row_overlaps_floor(pair.states, contacts, pair.a, row, pair.t_c)


def test_a_struck_ball_at_rest_between_resting_records_keeps_its_row_values(
    pair: Pair,
) -> None:
    """R5 f: the f5 pattern, a resting ball whose RESTING records leave a gap around the impact;
    it is within the contact tolerance at both rows, so it is on the floor by touch."""
    pair = pair.at_height(pair.b, (3e-4, 5e-4))
    contacts = pair.contacts(
        (pair.b, K - 5.0, K - 0.6, ContactKind.RESTING),
        (pair.b, K + 2.6, K + 20.0, ContactKind.RESTING),
    )
    for row in (K, K + 1):
        assert not row_overlaps_floor(pair.states, contacts, pair.b, row, pair.t_c)
        assert row_on_floor(pair.states, contacts, pair.b, row, pair.t_c, CONTACT_TOL)
    v_minus, v_plus = velocities(pair, contacts, pair.b)
    assert np.array_equal(v_minus, pair.states.velocity[K, pair.b])
    assert np.array_equal(v_plus, pair.states.velocity[K + 1, pair.b])


def test_a_tilted_box_resting_on_an_edge_touches_the_floor(pair: Pair) -> None:
    """R5 g: the gap is the geometric distance, not z - h_z / 2 (a box on an edge sits lower)."""
    edge, clearance = 0.2, 1e-3
    half = math.radians(45.0) / 2.0
    tilt = np.array([math.cos(half), math.sin(half), 0.0, 0.0])
    z = edge / math.sqrt(2.0) + clearance
    primitive = np.array(pair.states.primitive, copy=True)
    primitive[pair.b] = PrimitiveKind.BOX.value
    states = dataclasses.replace(pair.states, primitive=primitive)
    states = edit_object(
        states,
        pair.b,
        slice(K, K + 2),
        size=np.full((2, 3), edge),
        quaternion=np.repeat(tilt[None], 2, axis=0),
        position=pair._xy(pair.b, np.full(2, z)),
    )
    assert z - edge / 2.0 > CONTACT_TOL
    assert floor_gap(states, pair.b, K) == pytest.approx(clearance, abs=1e-9)
    contacts = pair.contacts()
    assert row_on_floor(states, contacts, pair.b, K, pair.t_c, CONTACT_TOL)


def test_a_ball_beyond_the_tolerance_without_a_record_is_extrapolated(pair: Pair) -> None:
    """R5 h: 3 mm above the floor and no record: flight."""
    pair = pair.at_height(pair.a, (3e-3, 3e-3))
    contacts = pair.contacts()
    assert floor_gap(pair.states, pair.a, K + 1) == pytest.approx(3e-3, abs=1e-12)
    v_minus, v_plus = velocities(pair, contacts, pair.a)
    assert np.allclose(v_minus, pair.v_minus[0], atol=TOL, rtol=0.0)
    assert np.allclose(v_plus, pair.v_plus[0], atol=TOL, rtol=0.0)


def test_the_tolerance_comes_from_the_contact_settings(ds: Any, perturbations: Any) -> None:
    """R2: the plugin's tolerance is ``[dataset.contacts].tol_m``, not the parameter default."""
    tol = 3.0 * CONTACT_TOL
    contacts = ds.contacts.model_copy(update={"tol_m": tol})
    plugins = instantiate(perturbations, ds.model_copy(update={"contacts": contacts}))
    assert plugins["r3_energy_gain"].params.contact_tol_m == tol


@pytest.mark.parametrize(
    ("gap", "vz", "dt", "stops"),
    [
        (0.01, -1.0, -0.03, True),
        (0.189, -2.29, -0.85 / 24.0, False),
        (0.004, 0.0, 0.03, True),
    ],
)
def test_a_flight_row_that_could_reach_the_floor_before_t_c_stops(
    gap: float, vz: float, dt: float, stops: bool
) -> None:
    """I11: the stop fires on an extrapolated row whose gap is within free-fall reach of t_c."""
    assert (floor_reach(vz, dt, 9.81) >= gap) is stops
