"""The config files split out of a single file hold exactly what it held: the same tables and
values, and the same non-blank lines, comments included (with the config paths they name moved to
the new layout). The originals, as of 5c62128, are in ``data/pre_split_configs.json``."""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
ORIGINALS = json.loads(
    (Path(__file__).resolve().parent / "data" / "pre_split_configs.json").read_text("utf-8")
)

SPLITS = {
    "agreement.toml": ("studies/human_agreement.toml", "datasets/videophy.toml"),
    "diagnosis.toml": (
        "studies/diagnosis.toml",
        "datasets/physics_iq.toml",
        "datasets/physics_iq_views.toml",
        "datasets/objectron_static.toml",
        "datasets/pisa_real.toml",
    ),
}

MOVED_PATHS = {"(configs/check_scope.toml)": "(configs/verifier/check_scope.toml)"}


def _with_moved_paths(text: str) -> str:
    for old, new in MOVED_PATHS.items():
        text = text.replace(old, new)
    return text


def _non_blank_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


@pytest.mark.parametrize("original", sorted(SPLITS))
def test_the_pieces_hold_the_same_tables(original: str) -> None:
    merged: dict[str, object] = {}
    for piece in SPLITS[original]:
        tables = tomllib.loads((CONFIGS / piece).read_text(encoding="utf-8"))
        assert not merged.keys() & tables.keys()
        merged.update(tables)
    assert merged == tomllib.loads(ORIGINALS[original])


@pytest.mark.parametrize("original", sorted(SPLITS))
def test_the_pieces_hold_the_same_lines(original: str) -> None:
    pieces = "\n".join((CONFIGS / piece).read_text(encoding="utf-8") for piece in SPLITS[original])
    old = _with_moved_paths(ORIGINALS[original])
    assert Counter(_non_blank_lines(pieces)) == Counter(_non_blank_lines(old))
