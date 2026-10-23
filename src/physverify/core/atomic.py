"""Atomic file writes: readers see either the previous file or the complete new one."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path


def write_atomic(path: Path, write: Callable[[Path], object]) -> None:
    """``write`` fills a sibling temp file that then replaces ``path``: readers never see a
    half-written file, and the temp file is removed on any failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(temp)
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    write_atomic(path, lambda temp: temp.write_text(text, encoding="utf-8"))


def write_bytes_atomic(path: Path, data: bytes) -> None:
    write_atomic(path, lambda temp: temp.write_bytes(data))
