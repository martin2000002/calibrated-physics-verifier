"""Option parsing and context lookups shared by the subcommands (not a plugin module)."""

from __future__ import annotations

from physverify.app.context import AppContext
from physverify.calibration.settings import active_checks_hash
from physverify.core.errors import UsageError


def choice[T: str](value: str, known: tuple[T, ...], option: str) -> T:
    for candidate in known:
        if value == candidate:
            return candidate
    raise UsageError(f"{option} must be one of {list(known)}, got {value!r}")


def context_checks_hash(context: AppContext) -> str:
    """The ``checks_hash`` of the context's check registry under its ``[checks]`` overrides."""
    return active_checks_hash(context.settings, context.registry("checks"))
