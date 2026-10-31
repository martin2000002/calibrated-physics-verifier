"""Shared contact code keeps its behaviour: the moved reference speed, the Bonferroni switch
of the impact detector and the split of proximity records into intervals and records."""

from __future__ import annotations

from dataclasses import replace

import contact_scenes as cs
import numpy as np
import pytest
from test_impacts import ALPHA, ROWS, TEST_SIGMA

from physverify.diagnosis import normalizers
from physverify.extraction.providers._impacts import Impact, track_impacts
from physverify.extraction.providers._proximity import ContactScene, ProximityParams
from physverify.physics import kinematics
from physverify.physics.tracks import NoiseScale
from physverify.schema.clip import SamplingSpec, resolve_sampling

sc = cs.sc
CASES = {
    "bounce": lambda: sc.bounce(e=0.8, impact_frame=15.3, pitch_deg=25.0),
    "parabola": lambda: sc.parabola(pitch_deg=25.0),
    "teleport": lambda: sc.teleport(pitch_deg=25.0),
    "collision": lambda: cs.collision_over_floor(),
    "noisy_bounce": lambda: noisy(sc.bounce(e=0.8, impact_frame=15.3, pitch_deg=25.0)),
}


def noisy(case: sc.Case) -> sc.Case:
    return replace(case, states=cs.jittered(case.states, 2.0 * TEST_SIGMA, seed=3))


def impacts(case: sc.Case, bonferroni: bool) -> list[list[Impact]]:
    states = case.states
    noise = NoiseScale(np.full(3, TEST_SIGMA), known=True)
    return [
        track_impacts(states.t, states.position[:, n], noise, ROWS, ALPHA, bonferroni=bonferroni)
        for n in range(states.n_objects)
    ]


def key(impact: Impact) -> tuple[int, float]:
    return impact.row, impact.time


def test_reference_speed_is_one_function() -> None:
    assert normalizers.reference_speed is kinematics.reference_speed
    assert "reference_speed" in normalizers.__all__


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_default_is_the_bonferroni_detector(name: str) -> None:
    case = CASES[name]()
    states = case.states
    noise = NoiseScale(np.full(3, TEST_SIGMA), known=True)
    for n in range(states.n_objects):
        default = track_impacts(states.t, states.position[:, n], noise, ROWS, ALPHA)
        explicit = impacts(case, True)[n]
        assert [key(i) for i in default] == [key(i) for i in explicit]


@pytest.mark.parametrize("name", sorted(CASES))
def test_shape_only_finds_a_superset_of_rows(name: str) -> None:
    case = CASES[name]()
    strict, loose = impacts(case, True), impacts(case, False)
    for kept, found in zip(strict, loose, strict=True):
        assert {i.row for i in kept} <= {i.row for i in found}


@pytest.mark.parametrize("name", sorted(CASES))
def test_records_are_the_records_of_the_intervals(name: str) -> None:
    case = CASES[name]()
    spf = resolve_sampling(case.clip, SamplingSpec()).seconds_per_frame
    scene = ContactScene(case.states, ProximityParams(), spf, split_on_point_count=True)
    whole = list(scene.records())
    split = [scene.record(pair, interval) for pair, interval in scene.intervals()]
    assert [(r.object_a, r.object_b, r.t_start, r.t_end, r.kind) for r in whole] == [
        (r.object_a, r.object_b, r.t_start, r.t_end, r.kind) for r in split
    ]
    for a, b in zip(whole, split, strict=True):
        np.testing.assert_array_equal(a.points, b.points)
        assert a.confidence == b.confidence or (np.isnan(a.confidence) and np.isnan(b.confidence))
