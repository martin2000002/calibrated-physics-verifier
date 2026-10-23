"""Reading JSON files with one error shape: the path and the reason, as the caller's error type."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ValidationError

from physverify.core.errors import PhysverifyError

ErrorFactory = Callable[[str], Exception]


def loads(text: str, path: Path, error: ErrorFactory = PhysverifyError) -> object:
    """The JSON value of ``text`` read from ``path``; invalid JSON raises ``error``."""
    try:
        data: object = json.loads(text)
    except ValueError as exc:
        raise error(f"{path} is not valid JSON: {exc}") from exc
    return data


def read_json(path: Path, error: ErrorFactory = PhysverifyError) -> object:
    """The JSON value in ``path``; an unreadable file or invalid JSON raises ``error``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise error(f"cannot read {path}: {exc}") from exc
    return loads(text, path, error)


def read_model[M: BaseModel](
    model: type[M], path: Path, error: ErrorFactory = PhysverifyError
) -> M:
    """``model`` validated from the JSON in ``path``; a failure raises ``error``."""
    data = read_json(path, error)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise error(f"{path} is not a valid {model.__name__}: {exc}") from exc
