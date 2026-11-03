"""The provisional threshold table names the committed file and table its taus come from."""

from __future__ import annotations

import tomllib
from pathlib import Path

from physverify.verification.thresholding import PROVISIONAL_SOURCE

REPO = Path(__file__).resolve().parents[2]


def test_the_provisional_source_is_the_committed_thresholds_table() -> None:
    path, table = PROVISIONAL_SOURCE.split(" ", 1)
    keys = table.strip("[]").split(".")
    found = tomllib.loads((REPO / path).read_text(encoding="utf-8"))
    for key in keys:
        found = found[key]
    assert found
