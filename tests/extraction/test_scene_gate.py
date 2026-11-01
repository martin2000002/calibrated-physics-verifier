"""The geometric gate of ``contacts/kinematic_scene``: when switched on, an environment rest needs
the body within z(alpha / 2) sigma_gap of its scene plane, sigma_gap from the profile's gap entry.
It is off by default."""

from __future__ import annotations

import math

import contact_scenes as cs
import pytest
import scene_contact_cases as cases

from physverify.extraction.providers._scene_plane import GapError
from physverify.schema.contacts import ContactKind
from physverify.schema.error_profile import SUPPORT_GAP_REL
from physverify.schema.stages import Stage

UNKNOWN = GapError.of(None)
GATE = {"env_gap_gate": True}


@pytest.mark.parametrize(("share", "rests"), [(0.9, True), (1.1, False)])
def test_a_body_rests_on_the_environment_only_within_the_gate(share: float, rests: bool) -> None:
    case = cases.boxes_at([0.0, 0.0, share * cases.gate()])
    records = cases.env_records(cases.outcome(case, **GATE).records, "box2")
    assert bool(records) is rests


def test_the_gate_scales_with_the_measured_gap_error() -> None:
    case = cases.boxes_at([0.0, 0.0, 1.1 * cases.gate()])
    wider = GapError(2.0 * cases.GAPS.relative)
    assert cases.env_records(cases.outcome(case, gaps=wider, **GATE).records, "box2")


def test_a_still_body_in_the_air_is_not_resting_while_the_floor_ones_are() -> None:
    case = cases.boxes_at([0.0, 0.0, cases.LENGTH])
    records = cases.env_records(cases.outcome(case, **GATE).records)
    assert sorted(r.object_a for r in records) == ["box0", "box1"]
    assert {r.kind for r in records} == {ContactKind.RESTING}


def test_the_floating_body_still_sets_no_plane_under_the_others() -> None:
    case = cases.boxes_at([0.0, 0.0, cases.LENGTH])
    found = cases.outcome(case, **GATE)
    assert [p.source for p in found.planes if p] == [1, 0, 0]


def test_an_unknown_gap_error_tests_no_environment_rest_and_says_so() -> None:
    case = cases.stacked_boxes()
    floor = cs.sc.build("floor", [cases.box("box2", 3 * cases.SPACING, 0.0)], [], 25.0).states
    states = cs.joined(case.states, floor)
    known = cases.outcome(case, states=states, **GATE)
    unknown = cases.outcome(case, states=states, gaps=UNKNOWN, **GATE)
    assert known.gap_known and not unknown.gap_known
    assert cases.env_records(known.records)
    assert cases.env_records(unknown.records) == []
    assert cases.pair_records(unknown.records) == cases.pair_records(known.records)


def test_environment_impacts_do_not_need_the_gap_error() -> None:
    case = cs.bounce_on_floor()
    records = cases.env_records(cases.outcome(case, gaps=UNKNOWN, **GATE).records)
    assert {r.kind for r in records} == {ContactKind.IMPACT}


def test_the_provider_reads_the_lift_gap_entry() -> None:
    case = cases.boxes_at([0.0, 0.0])
    entries = cases.measured_entries()
    assert cases.env_records(cases.scene_table(case, entries, **GATE).records)
    del entries[(Stage.LIFT, SUPPORT_GAP_REL)]
    assert cases.env_records(cases.scene_table(case, entries, **GATE).records) == []


def test_without_the_gate_a_still_body_in_the_air_rests_on_its_plane() -> None:
    case = cases.boxes_at([0.0, 0.0, cases.LENGTH])
    records = cases.env_records(cases.outcome(case).records)
    assert sorted(r.object_a for r in records) == ["box0", "box1", "box2"]


def test_without_the_gate_an_unknown_gap_error_keeps_the_environment_rest() -> None:
    case = cases.boxes_at([0.0, 0.0])
    found = cases.outcome(case, gaps=UNKNOWN)
    assert not found.gap_known
    records = cases.env_records(found.records)
    assert {r.kind for r in records} == {ContactKind.RESTING}
    assert all(math.isnan(r.confidence) for r in records)
