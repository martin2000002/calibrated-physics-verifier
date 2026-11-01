"""The registered R5a (``FallShapeCheck``) reproduces, field by field within 1e-9, what the code
of commit 5140a4e wrote into ``golden/r5a_v1.json`` (``r5a_v1_golden_case.py``). In a degenerate
result (magnitude below 1e-9) every candidate ties up to float noise, so which one wins depends
on the platform's BLAS: its fields that only locate the winner are not pinned."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import pytest
from r5a_v1_golden_case import GOLDEN, SCENES, results

TOLERANCE = 1e-9
WINNER_FIELDS = frozenset({"frame_span", "mode", "split_frame", "windows"})


@pytest.fixture(scope="module")
def current() -> dict[str, list[Any]]:
    return results()


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_the_v1_results_do_not_move(current: dict[str, list[Any]], scene: str) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[scene]
    assert len(current[scene]) == len(golden)
    for got, want in zip(current[scene], golden, strict=True):
        assert_close(got, want, scene, unpinned=_unpinned(want))


@pytest.mark.parametrize("scene", sorted(s for s in SCENES if s.endswith("_jittered")))
def test_the_jittered_scenes_are_not_degenerate(scene: str) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[scene]
    assert all(r["magnitude"] > 1e-3 for r in golden)


def _unpinned(result: Mapping[str, Any]) -> frozenset[str]:
    magnitude = result.get("magnitude")
    degenerate = isinstance(magnitude, float) and abs(magnitude) < TOLERANCE
    return WINNER_FIELDS if degenerate else frozenset()


def assert_close(got: Any, want: Any, where: str, unpinned: frozenset[str]) -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), where
        for key in want:
            if key not in unpinned:
                assert_close(got[key], want[key], f"{where}.{key}", unpinned)
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), where
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            assert_close(g, w, f"{where}[{i}]", unpinned)
    elif isinstance(want, float) and not isinstance(got, bool) and isinstance(got, int | float):
        assert math.isclose(got, want, rel_tol=TOLERANCE, abs_tol=TOLERANCE), (where, got, want)
    else:
        assert got == want, (where, got, want)
