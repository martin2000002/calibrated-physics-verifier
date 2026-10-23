"""Moving ``pixel_floor``, ``noise_scale``, ``NoiseScale`` and
``QUANTIZATION`` into ``physics/tracks.py`` does not move any contacts/kinematic table; the
goldens were written by 10e37ff with ``kinematic_golden_case.py``."""

from __future__ import annotations

import pytest
from contacts_golden_case import plain
from golden import same
from kinematic_golden_case import CASES, GOLDEN

from physverify.schema.contacts import ContactTable


@pytest.mark.parametrize("name", sorted(CASES))
def test_t15_the_kinematic_tables_do_not_move(name: str) -> None:
    golden = ContactTable.load_json(GOLDEN / f"{name}.json")
    assert same(plain(CASES[name]()), plain(golden))


def test_t15_the_goldens_hold_kinematic_impacts() -> None:
    sources = {
        record.source
        for name in CASES
        for record in ContactTable.load_json(GOLDEN / f"{name}.json").records
    }
    assert "kinematic" in sources
