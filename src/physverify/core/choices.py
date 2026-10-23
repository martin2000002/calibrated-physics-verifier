"""A string option restricted to a fixed set of values."""

from __future__ import annotations

from physverify.core.errors import UsageError


def choice[T: str](value: str, known: tuple[T, ...], option: str) -> T:
    """``value`` as one of ``known``, else a usage error naming ``option``."""
    for candidate in known:
        if value == candidate:
            return candidate
    raise UsageError(f"{option} must be one of {list(known)}, got {value!r}")
