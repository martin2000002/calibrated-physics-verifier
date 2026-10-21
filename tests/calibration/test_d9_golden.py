"""T14 (it1 audit d1): with sigma_pair_filter = "all" the fit and the devscore report are
the ones written before it1, except ``version`` and the fields it1 added. Floats match to a
relative 1e-9: the golden was written on another CPU, where BLAS-backed fits differ in the last
bits."""

from __future__ import annotations

import json
import math
from typing import Any

from d9_golden_case import GOLDEN, fitted

# Fields added by it1, with the value they take under "all"; any other difference fails.
ADDED_AT_ALL: dict[str, Any] = {
    "n_informative": None,
    "coverage_after_informative": [],
    "coverage_after_degenerate": [],
    "sigma_pair_filter": "all",
}
ADDED_ANY = ("folds_without_sigma_factor",)  # devscore count, new and not tied to the filter


def without_it1(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_it1(item)
            for key, item in value.items()
            if key != "version"
            and key not in ADDED_ANY
            and not (key in ADDED_AT_ALL and item == ADDED_AT_ALL[key])
        }
    if isinstance(value, list):
        return [without_it1(item) for item in value]
    return value


REL_TOL = 1e-9


def same(now: Any, before: Any) -> bool:
    if isinstance(now, float) and isinstance(before, float):
        return math.isclose(now, before, rel_tol=REL_TOL, abs_tol=0.0) or now == before
    if isinstance(now, dict) and isinstance(before, dict):
        return now.keys() == before.keys() and all(same(now[k], before[k]) for k in now)
    if isinstance(now, list) and isinstance(before, list):
        return len(now) == len(before) and all(same(a, b) for a, b in zip(now, before, strict=True))
    return bool(now == before)


def golden() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return loaded


def test_the_all_filter_reproduces_the_fit_before_it1() -> None:
    now, before = fitted(sigma_pair_filter="all"), golden()
    for part in ("two_level", "devscore"):
        assert same(without_it1(now[part]), without_it1(before[part])), part


def test_the_golden_case_is_sensitive_to_the_filter() -> None:
    nonzero = without_it1(fitted(sigma_pair_filter="nonzero")["two_level"])
    assert not same(nonzero, without_it1(golden()["two_level"]))
