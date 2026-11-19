"""The registered R5b (``GravityValueCheck``) reproduces, field by field, what the code of commit
4671f02 wrote into ``golden/r5b_v1.json`` (``r5b_v1_golden_case.py``), including the scenes a
variant treats differently (an unmarked roll and a gliding ball beside a resting box). Floats
match within ``golden.same``'s tolerance: the fits differ in the last bits across CPUs."""

from __future__ import annotations

import json
from typing import Any

import pytest
from golden import same
from r5b_v1_golden_case import GOLDEN, SCENES, results


@pytest.fixture(scope="module")
def current() -> dict[str, list[Any]]:
    return results()


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_the_v1_results_do_not_move(current: dict[str, list[Any]], scene: str) -> None:
    assert same(current[scene], json.loads(GOLDEN.read_text(encoding="utf-8"))[scene])


def test_the_golden_holds_every_scene_and_scores_the_unmarked_roll() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert sorted(golden) == sorted(SCENES)
    assert golden["unmarked_roll"][0]["magnitude"] == pytest.approx(1.0)
