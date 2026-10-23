"""The config files split out of a single file hold exactly what it held: the same tables and
values, and the same non-blank lines, comments included."""

from __future__ import annotations

import tomllib
from collections import Counter
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
DATA = Path(__file__).resolve().parent / "data"

SPLITS = {
    "pre_split_agreement.toml": ("studies/human_agreement.toml", "datasets/videophy.toml"),
    "pre_split_diagnosis.toml": (
        "studies/diagnosis.toml",
        "datasets/physics_iq.toml",
        "datasets/physics_iq_views.toml",
        "datasets/objectron_static.toml",
        "datasets/pisa_real.toml",
    ),
}


def _non_blank_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


@pytest.mark.parametrize("original", sorted(SPLITS))
def test_the_pieces_hold_the_same_tables(original: str) -> None:
    merged: dict[str, object] = {}
    for piece in SPLITS[original]:
        tables = tomllib.loads((CONFIGS / piece).read_text(encoding="utf-8"))
        assert not merged.keys() & tables.keys()
        merged.update(tables)
    assert merged == tomllib.loads((DATA / original).read_text(encoding="utf-8"))


@pytest.mark.parametrize("original", sorted(SPLITS))
def test_the_pieces_hold_the_same_lines(original: str) -> None:
    pieces = "\n".join((CONFIGS / piece).read_text(encoding="utf-8") for piece in SPLITS[original])
    old = (DATA / original).read_text(encoding="utf-8")
    assert Counter(_non_blank_lines(pieces)) == Counter(_non_blank_lines(old))
