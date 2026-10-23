"""Cell, table and section formatting shared by the diagnosis markdown reports."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum

from physverify.core.markdown import number, table
from physverify.diagnosis.report import MISSING


def cell_text(value: object) -> str:
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return number(value, ".4g", MISSING, "nonfinite")
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, tuple):
        return ", ".join(cell_text(item) for item in value) or "-"
    return str(value).replace("|", "\\|")


def text_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    return [*table(headers, ([cell_text(value) for value in cells] for cells in rows)), ""]


def section(title: str, lines: Sequence[str]) -> list[str]:
    return [f"## {title}", "", *lines] if lines else []
