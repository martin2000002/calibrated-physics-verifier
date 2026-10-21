"""it3 T1: moving the proximity helpers into ``_proximity.py`` changes no table of
``contacts/proximity`` (golden written by 2a762ad, ``contacts_golden_case.py``)."""

from __future__ import annotations

import pytest
from contacts_golden_case import CASES, GOLDEN, proximity_table

from physverify.extraction.providers.contacts_proximity import ProximityContacts
from physverify.schema.contacts import ContactTable


@pytest.mark.parametrize("case", sorted(CASES))
def test_t1_proximity_tables_equal_the_golden(case: str) -> None:
    golden = ContactTable.load_json(GOLDEN / f"{case}.json")
    assert proximity_table(CASES[case]()) == golden


def test_t1_the_proximity_provider_keeps_version_one() -> None:
    assert ProximityContacts.version == "1"
