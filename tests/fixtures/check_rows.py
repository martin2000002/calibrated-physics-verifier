"""checks.jsonl files as verify writes them, for the H200 scripts that inspect a verify run."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from records import check_result

from physverify.schema.io import write_jsonl

UNCACHED = "MissingInputError: a heavy stage is not cached"


def write_check_rows(path: Path, *, error: str | None = None) -> Path:
    """Two rows of one clip; with ``error`` the second carries it like a clip without inputs."""
    clean = check_result()
    rows = [clean]
    if error is not None:
        rows.append(dataclasses.replace(clean, check_id="R1", details={"error": error}))
    write_jsonl(path, rows)
    return path
