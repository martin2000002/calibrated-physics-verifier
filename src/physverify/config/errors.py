"""One line per pydantic error of a configuration section, without the documentation URLs."""

from __future__ import annotations

from pydantic import ValidationError
from pydantic_core import ErrorDetails

_VALUE_ERROR_PREFIX = "Value error, "


def describe_errors(section: str, exc: ValidationError) -> str:
    """``[section] key: message (got value)`` per error, e.g. ``[runtime] threads: Input should
    be a valid integer, unable to parse string as an integer (got 'abc')``."""
    return "\n".join(_line(section, error) for error in exc.errors(include_url=False))


def _line(section: str, error: ErrorDetails) -> str:
    key = ".".join(str(part) for part in error["loc"])
    message = error["msg"].removeprefix(_VALUE_ERROR_PREFIX)
    if not key:
        return f"[{section}] {message}"
    got = "" if error["type"] == "missing" else f" (got {error['input']!r})"
    return f"[{section}] {key}: {message}{got}"
