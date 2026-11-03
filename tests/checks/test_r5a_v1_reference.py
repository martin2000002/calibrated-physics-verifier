"""The registered R5a (``FallShapeCheck``) reproduces, field by field within the golden tolerance
(``golden.same``), what the code of commit 5140a4e wrote into ``golden/r5a_v1.json``
(``r5a_v1_golden_case.py``). In a degenerate result (magnitude below 1e-9) every candidate ties up
to float noise, so which one wins depends on the platform's BLAS: its fields that only locate the
winner are not pinned."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest
from golden import same
from r5a_v1_golden_case import GOLDEN, SCENES, results

DEGENERATE = 1e-9
"""A magnitude below it is float noise: every candidate ties."""
WINNER_FIELDS = frozenset({"frame_span", "mode", "split_frame", "windows"})


@pytest.fixture(scope="module")
def current() -> dict[str, list[Any]]:
    return results()


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_the_v1_results_do_not_move(current: dict[str, list[Any]], scene: str) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[scene]
    assert len(current[scene]) == len(golden)
    for got, want in zip(current[scene], golden, strict=True):
        unpinned = _unpinned(want)
        assert same(_pinned(got, unpinned), _pinned(want, unpinned)), (scene, got, want)


@pytest.mark.parametrize("scene", sorted(s for s in SCENES if s.endswith("_jittered")))
def test_the_jittered_scenes_are_not_degenerate(scene: str) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[scene]
    assert all(r["magnitude"] > 1e-3 for r in golden)


def _unpinned(result: Mapping[str, Any]) -> frozenset[str]:
    magnitude = result.get("magnitude")
    degenerate = isinstance(magnitude, float) and abs(magnitude) < DEGENERATE
    return WINNER_FIELDS if degenerate else frozenset()


def _pinned(value: Any, unpinned: frozenset[str]) -> Any:
    """``value`` without the ``unpinned`` keys, at any depth."""
    if isinstance(value, Mapping):
        return {k: _pinned(v, unpinned) for k, v in value.items() if k not in unpinned}
    if isinstance(value, list):
        return [_pinned(v, unpinned) for v in value]
    return value
