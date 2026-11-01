"""Version 3 of ``contacts/kinematic_scene`` with the 3D velocity rest and the gap gate is version
2; its defaults are pinned by their own goldens (``scene_golden_case.py``)."""

from __future__ import annotations

import pytest
from contacts_golden_case import plain
from golden import same
from scene_golden_case import CASES, GOLDEN, V2_EQUIVALENT, table

from physverify.schema.contacts import ContactTable


def golden(version: str, name: str) -> ContactTable:
    return ContactTable.load_json(GOLDEN / version / f"{name}.json")


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_velocity_rest_with_the_gate_is_version_2(name: str) -> None:
    assert same(plain(table(CASES[name](), **V2_EQUIVALENT)), plain(golden("v2", name)))


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_version_3_tables_do_not_move(name: str) -> None:
    assert same(plain(table(CASES[name]())), plain(golden("v3", name)))


def test_the_image_rest_without_the_gate_moves_only_some_tables() -> None:
    moved = {
        name for name in CASES if not same(plain(golden("v2", name)), plain(golden("v3", name)))
    }
    assert moved and moved != set(CASES)
