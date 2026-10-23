"""Markdown table lines and number cells shared by the reports."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Literal

MissingWhen = Literal["none", "nan", "nonfinite"]


def number(value: float | None, spec: str, missing: str, when: MissingWhen = "none") -> str:
    """``value`` in format ``spec``, or ``missing`` when it is None (or NaN / non-finite)."""
    if value is None:
        return missing
    if (when == "nan" and math.isnan(value)) or (when == "nonfinite" and not math.isfinite(value)):
        return missing
    return format(value, spec)


def row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    return [row(headers), "|" + "---|" * len(headers), *(row(cells) for cells in rows)]
