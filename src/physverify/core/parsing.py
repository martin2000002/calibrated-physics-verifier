"""Lenient parsing of text cells."""

from __future__ import annotations

import math


def to_float(text: str | None) -> float:
    """``text`` as a float; NaN when it is missing, empty or not a number."""
    try:
        return float(text) if text not in (None, "") else math.nan
    except ValueError:
        return math.nan
