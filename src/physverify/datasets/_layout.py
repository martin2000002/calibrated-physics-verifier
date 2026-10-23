"""Where the real datasets keep their clips, and their hand-written query annotations."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import RootModel, ValidationError

from physverify.core.errors import ConfigError
from physverify.core.local_manifest import QueryList, to_queries
from physverify.schema.clip import ObjectQuery


def relative_to_data(path: Path, data_dir: Path) -> str:
    """``ClipManifest.video_path``: relative to ``data_dir`` when inside it, else absolute."""
    return str(path.relative_to(data_dir)) if path.is_relative_to(data_dir) else str(path)


def resolve_data_path(video_path: str, data_dir: Path) -> Path:
    path = Path(video_path)
    return path if path.is_absolute() else data_dir / path


class _Annotations(RootModel[dict[str, QueryList]]):
    pass


def load_annotations(path: Path) -> dict[str, tuple[ObjectQuery, ...]]:
    """``{clip_id or scenario: queries}`` from a hand-written annotation file (empty if absent)."""
    if not path.is_file():
        return {}
    try:
        table = _Annotations.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ConfigError(f"{path} is invalid:\n{exc}") from exc
    return {key: to_queries(entries) for key, entries in table.root.items()}
