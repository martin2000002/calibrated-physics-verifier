"""SHA-256 hex digests of files and text."""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_BYTES = 1 << 20


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_if_file(path: Path) -> str | None:
    return file_sha256(path) if path.is_file() else None


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
