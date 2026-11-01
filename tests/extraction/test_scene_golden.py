"""Version 4 of ``contacts/kinematic_scene`` with the 3D velocity impacts and the proximity pair
rest is version 3, and with the 3D velocity rest and the gap gate as well, version 2; its defaults
are pinned by their own goldens (``scene_golden_case.py``)."""

from __future__ import annotations

import pytest
from contacts_golden_case import plain
from golden import same
from scene_golden_case import CASES, GOLDEN, V2_EQUIVALENT, V3_EQUIVALENT, table

from physverify.schema.contacts import ContactTable


def golden(version: str, name: str) -> ContactTable:
    return ContactTable.load_json(GOLDEN / version / f"{name}.json")


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_velocity_rest_with_the_gate_is_version_2(name: str) -> None:
    assert same(plain(table(CASES[name](), **V2_EQUIVALENT)), plain(golden("v2", name)))


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_velocity_impacts_and_the_proximity_pairs_are_version_3(name: str) -> None:
    assert same(plain(table(CASES[name](), **V3_EQUIVALENT)), plain(golden("v3", name)))


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_version_4_tables_do_not_move(name: str) -> None:
    assert same(plain(table(CASES[name]())), plain(golden("v4", name)))


@pytest.mark.parametrize(("old", "new"), [("v2", "v3"), ("v3", "v4")])
def test_each_version_moves_only_some_tables(old: str, new: str) -> None:
    moved = {name for name in CASES if not same(plain(golden(old, name)), plain(golden(new, name)))}
    assert moved and moved != set(CASES)
