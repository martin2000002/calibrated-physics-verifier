"""The config files split out of a single file hold exactly what it held: the same tables and
values. The originals, as of 5c62128, are in ``data/pre_split_configs.json``; keys added after
the split are declared in ``ADDED_SINCE_SPLIT``."""

from __future__ import annotations

import json
import tomllib
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
ADDED_SINCE_SPLIT: dict[str, dict[str, frozenset[str]]] = {
    "agreement.toml": {"agreement": frozenset({"min_cut_side_fraction"})},
}


def _without_added(original: str, merged: dict[str, object]) -> dict[str, object]:
    found = dict(merged)
    for table, keys in ADDED_SINCE_SPLIT.get(original, {}).items():
        section = dict(found[table])  # type: ignore[call-overload]
        assert keys <= section.keys(), f"[{table}] lacks {sorted(keys - section.keys())}"
        found[table] = {k: v for k, v in section.items() if k not in keys}
    return found


@pytest.mark.parametrize("original", sorted(SPLITS))
def test_the_pieces_hold_the_same_tables(original: str) -> None:
    merged: dict[str, object] = {}
    for piece in SPLITS[original]:
        tables = tomllib.loads((CONFIGS / piece).read_text(encoding="utf-8"))
        assert not merged.keys() & tables.keys()
        merged.update(tables)
    assert _without_added(original, merged) == tomllib.loads(ORIGINALS[original])
