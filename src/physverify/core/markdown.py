"""Markdown table lines and number cells shared by the reports."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

MissingWhen = Literal["none", "nan", "nonfinite"]


def number(value: float | None, spec: str, missing: str, when: MissingWhen = "none") -> str:
    """``value`` in format ``spec``, or ``missing`` when it is None (or NaN / non-finite)."""
    if value is None:
        return missing
    if (when == "nan" and math.isnan(value)) or (when == "nonfinite" and not math.isfinite(value)):
        return missing
    return format(value, spec)


@dataclass(frozen=True)
class NumberFormat:
    """A named ``number`` preset, so every report of one style formats a cell the same way."""

    spec: str
    missing: str = "-"
    when: MissingWhen = "none"

    def __call__(self, value: float | None) -> str:
        return number(value, self.spec, self.missing, self.when)


SIG4 = NumberFormat(".4g")
SIG4_NAN = NumberFormat(".4g", when="nan")
SIG4_NAN_TEXT = NumberFormat(".4g", "nan", "nan")
SIG3_NAN = NumberFormat(".3g", when="nan")
SIG3_FINITE = NumberFormat(".3g", when="nonfinite")
FIXED3 = NumberFormat(".3f", "n/a")


def row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    return [row(headers), "|" + "---|" * len(headers), *(row(cells) for cells in rows)]
