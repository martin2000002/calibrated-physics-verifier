"""The measured per-stage error profile the providers read, when one has been written."""

from __future__ import annotations

from pathlib import Path

from physverify.schema.error_profile import ErrorProfile


def load_error_profile(path: Path) -> ErrorProfile | None:
    return ErrorProfile.load_json(path) if path.is_file() else None
