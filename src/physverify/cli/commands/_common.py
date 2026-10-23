"""Option parsing shared by the subcommands (not a plugin module)."""

from __future__ import annotations

from physverify.core.errors import UsageError


def choice[T: str](value: str, known: tuple[T, ...], option: str) -> T:
    for candidate in known:
        if value == candidate:
            return candidate
    raise UsageError(f"{option} must be one of {list(known)}, got {value!r}")
