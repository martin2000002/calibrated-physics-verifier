"""Lenient float parsing of table cells."""

from __future__ import annotations

import math

import pytest

from physverify.core.parsing import to_float


@pytest.mark.parametrize("text", [None, "", "n/a"])
def test_a_missing_or_unparsable_cell_is_nan(text: str | None) -> None:
    assert math.isnan(to_float(text))


def test_a_number_parses() -> None:
    assert to_float("2.5") == 2.5
