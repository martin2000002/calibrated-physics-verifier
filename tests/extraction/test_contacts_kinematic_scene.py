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


def test_environment_points_sit_on_the_body_bottom_with_distance_zero() -> None:
    lift = 0.05 * cases.LENGTH
    case = cases.boxes_at([0.0, 0.0, lift])
    up, bottom = cases.world_plane(case, lift)
    found = cases.outcome(case)
    (record,) = cases.env_records(found.records, "box2")
    assert found.planes[2] is not None
    assert found.planes[2].height == pytest.approx(cases.world_plane(case)[1], abs=TOL)
    assert record.point_distances == pytest.approx(np.zeros(4), abs=TOL)
    assert cases.midway_offsets(record) == pytest.approx(np.full(4, bottom), abs=TOL)
    np.testing.assert_allclose(record.normal, up, atol=TOL)


def test_an_environment_impact_is_one_point_at_the_bottom_with_distance_zero() -> None:
    case = cs.bounce_on_floor()
    records = cases.env_records(cases.outcome(case).records)
    impacts = [r for r in records if r.kind is ContactKind.IMPACT]
    assert impacts
    for record in impacts:
        assert record.points.shape == (1, 3)
        assert record.point_distances == pytest.approx(np.zeros(1), abs=TOL)
        np.testing.assert_allclose(record.normal, cases.world_plane(case)[0], atol=TOL)


def test_records_are_sorted_and_marked_as_kinematic() -> None:
    records = cases.outcome(cs.bounce_on_floor()).records
    order = [(r.object_a, r.object_b, r.t_start) for r in records]
    assert order == sorted(order)
    assert {r.source for r in records} == {"kinematic_scene"}


# 9. The oracle is untouched


def test_the_provider_reads_the_track_lift_and_kinematics_entries_and_is_light() -> None:
    provider = KinematicSceneContacts
    assert (provider.stage, provider.name, provider.version) == (
        Stage.CONTACTS,
        "kinematic_scene",
        "3",
    )
    assert provider.requires == (Stage.TRACK, Stage.CAMERA, Stage.KINEMATICS)
    assert provider.error_profile_reads == (Stage.TRACK, Stage.LIFT, Stage.KINEMATICS)
    assert (provider.Params().rest, provider.Params().env_gap_gate) == ("image", False)
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
