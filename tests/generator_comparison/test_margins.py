from __future__ import annotations

import pytest

from physverify.core.errors import UsageError
from physverify.generator_comparison.margins import generator_means, margin, truncate

ROWS = [
    {"model_name": "A", "pc_pred": "3.6"},
    {"model_name": "A", "pc_pred": "3.6"},
    {"model_name": "B", "pc_pred": "3.86"},
    {"model_name": "B", "pc_pred": "3.86"},
    {"model_name": "C", "pc_pred": "1.0"},
    {"model_name": "B", "pc_pred": "not a number"},
    {"model_name": "A", "pc_pred": ""},
]


def test_truncation_never_rounds_up() -> None:
    assert truncate(0.13000000000000012) == 0.13
    assert truncate(0.1399) == 0.13
    assert truncate(0.05) == 0.05


def test_non_numeric_cells_are_skipped() -> None:
    means = generator_means(ROWS, "pc_pred", "model_name")
    assert {name: m.n for name, m in means.items()} == {"A": 2, "B": 2, "C": 1}
    assert means["B"].mean == pytest.approx(3.86)


def test_margin_is_half_the_spread_of_large_generators() -> None:
    derived = margin(ROWS, "pc_pred", "model_name", min_n=2)
    assert (derived.lo_generator, derived.hi_generator) == ("A", "B")
    assert derived.n_generators == 2
    assert derived.spread == pytest.approx(0.26)
    assert derived.value == 0.13


def test_margin_needs_two_generators() -> None:
    with pytest.raises(UsageError):
        margin(ROWS, "pc_pred", "model_name", min_n=3)
