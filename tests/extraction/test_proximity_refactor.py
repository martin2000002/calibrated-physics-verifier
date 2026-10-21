"""it3 T1: moving the proximity helpers into ``_proximity.py`` changes no table of
``contacts/proximity`` (golden written by 2a762ad, ``contacts_golden_case.py``), and
``contacts/kinematic`` with ``impacts = "none"`` and ``resting_split = "normal_and_count"`` gives
the same records."""

from __future__ import annotations

from dataclasses import replace

import contact_scenes as cs
import pytest
from contacts_golden_case import CASES, GOLDEN, proximity_table

from physverify.extraction.providers.contacts_kinematic import KinematicContacts
from physverify.extraction.providers.contacts_proximity import ProximityContacts
from physverify.schema.contacts import ContactTable


@pytest.mark.parametrize("case", sorted(CASES))
def test_t1_proximity_tables_equal_the_golden(case: str) -> None:
    golden = ContactTable.load_json(GOLDEN / f"{case}.json")
    assert proximity_table(CASES[case]()) == golden


def test_t1_the_proximity_provider_keeps_version_one() -> None:
    assert ProximityContacts.version == "1"


@pytest.mark.parametrize("case", sorted(CASES))
def test_t1_kinematic_without_impacts_and_with_the_count_split_is_proximity(case: str) -> None:
    golden = ContactTable.load_json(GOLDEN / f"{case}.json")
    provider = KinematicContacts(
        KinematicContacts.Params(impacts="none", resting_split="normal_and_count")
    )
    states = CASES[case]()
    unused = cs.scene_camera(cs.sc.stack())
    records = provider.records(states, unused, cs.sc.DT)
    assert replace(golden, records=records) == golden
