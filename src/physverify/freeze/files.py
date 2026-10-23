"""Reading the files a freeze records, with a failure as a QueryError."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from physverify.core.digests import file_sha256
from physverify.freeze.errors import QueryError


def readable_sha256(path: Path) -> str:
    try:
        return file_sha256(path)
    except OSError as exc:
        raise QueryError(f"cannot read {path}: {exc}") from exc


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QueryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise QueryError(f"{path} is not a JSON object")
    return data


def field(data: Mapping[str, Any], key: str) -> str:
    value: Any = data
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise QueryError(f"no {key} in the JSON")
        value = value[part]
    if isinstance(value, (dict, list)) or value is None:
        raise QueryError(f"{key} is not a single value")
    return str(value)
