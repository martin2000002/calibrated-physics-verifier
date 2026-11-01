"""The registered R5a (``FallShapeCheck``) reproduces exactly, field by field, what the code of
commit 5140a4e wrote into ``golden/r5a_v1.json`` (``r5a_v1_golden_case.py``)."""

from __future__ import annotations

import json

import pytest
from r5a_v1_golden_case import GOLDEN, SCENES, results


@pytest.fixture(scope="module")
def current() -> dict[str, list[object]]:
    return results()


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_the_v1_results_do_not_move(current: dict[str, list[object]], scene: str) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert current[scene] == golden[scene]
