"""``contacts/kinematic_scene`` as a plugin: its point convention, its records and the oracle
plan it must stay out of."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import scene_contact_cases as cases
from settings_factory import SettingsFactory
from wiring import resolve_plan

from physverify.diagnosis.chain_variants import oracle_variant
from physverify.extraction._base import StageProvider
from physverify.extraction.providers.contacts_kinematic_scene import KinematicSceneContacts
from physverify.schema.contacts import ContactKind
from physverify.schema.stages import Stage

cs = cases.cs
Providers = Mapping[str, type[StageProvider]]
TOL = 1e-9


# 8. Point convention


def test_environment_points_sit_midway_so_the_plane_offset_is_the_scene_height() -> None:
    case = cs.bounce_on_floor()
    found = cases.outcome(case)
    records = cases.env_records(found.records)
    assert {r.kind for r in records} == {ContactKind.IMPACT, ContactKind.RESTING}
    ids = list(case.states.object_ids)
    for record in records:
        plane = found.planes[ids.index(record.object_a)]
        assert plane is not None
        offsets = cases.midway_offsets(record)
        assert offsets == pytest.approx(np.full(len(offsets), plane.height), abs=TOL)
        np.testing.assert_allclose(record.normal, cases.world_plane(case)[0], atol=TOL)


def test_records_are_sorted_and_marked_as_kinematic() -> None:
    records = cases.outcome(cs.bounce_on_floor()).records
    order = [(r.object_a, r.object_b, r.t_start) for r in records]
    assert order == sorted(order)
    assert {r.source for r in records} == {"kinematic_scene"}


# 9. The oracle is untouched


def test_the_provider_reads_the_lift_and_kinematics_entries_and_is_light() -> None:
    provider = KinematicSceneContacts
    assert (provider.stage, provider.name, provider.version) == (
        Stage.CONTACTS,
        "kinematic_scene",
        "1",
    )
    assert provider.error_profile_reads == (Stage.LIFT, Stage.KINEMATICS)
    assert not provider.heavy and not provider.is_ground_truth


def test_the_precise_oracle_and_perceived_plans_do_not_use_the_provider(
    make_settings: SettingsFactory, real_providers: Providers
) -> None:
    settings = make_settings(profile="precise")
    oracle = resolve_plan(
        settings, real_providers, dataset="kubric_synthetic", gt_stages=oracle_variant().gt_order
    )
    perceived = resolve_plan(settings, real_providers, dataset="kubric_synthetic")
    assert oracle.names[Stage.CONTACTS] == "ground_truth"
    assert perceived.names[Stage.CONTACTS] != KinematicSceneContacts.name
