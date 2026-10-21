"""it4 T8 and T14: the v1 R4_jump (``ContinuityCheck``, out of the registry) and R4_size with
``reference = "self"`` reproduce, field by field, what the code before it4 (10e37ff) wrote into
``golden/r4_v1.json`` (``r4_v1_golden_case.py``)."""

from __future__ import annotations

import json

import pytest
from golden import same
from r4_v1_golden_case import GOLDEN, SCENES, results


@pytest.fixture(scope="module")
def current() -> dict[str, list[object]]:
    return results({"reference": "self"})


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_t8_t14_the_v1_results_do_not_move(current: dict[str, list[object]], scene: str) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert same(current[scene], golden[scene])
