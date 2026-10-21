"""Comparison of results against golden files written on another machine: floats match to a
relative 1e-9 (or 1e-12 absolute, for values that are rounding noise around 0), because BLAS-backed
fits and finite differences differ in the last bits across CPUs."""

from __future__ import annotations

import math
from typing import Any

REL_TOL = 1e-9
ABS_TOL = 1e-12


def same(now: Any, before: Any) -> bool:
    if isinstance(now, float) and isinstance(before, float):
        return now == before or math.isclose(now, before, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    if isinstance(now, dict) and isinstance(before, dict):
        return now.keys() == before.keys() and all(same(now[k], before[k]) for k in now)
    if isinstance(now, list) and isinstance(before, list):
        return len(now) == len(before) and all(same(a, b) for a, b in zip(now, before, strict=True))
    return bool(now == before)
