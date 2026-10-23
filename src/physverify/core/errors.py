"""Exception hierarchy of the host code; each class carries its process exit code."""

from __future__ import annotations

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_LOCK_BUSY = 75


class PhysverifyError(Exception):
    """Base error; the CLI prints the message and exits with ``exit_code``."""

    exit_code = EXIT_ERROR


class UsageError(PhysverifyError):
    exit_code = EXIT_USAGE


class ConfigError(PhysverifyError):
    exit_code = EXIT_USAGE


class ResourceBudgetError(PhysverifyError):
    """A request would exceed a configured memory budget (for example max_frame_bytes)."""

    exit_code = EXIT_USAGE


class LockBusyError(PhysverifyError):
    """The heavy-job lock is held by another process and the timeout expired."""

    exit_code = EXIT_LOCK_BUSY


class TimingError(PhysverifyError):
    pass


class MissingInputError(PhysverifyError):
    """A required upstream artifact or input file is not available."""


class PluginError(PhysverifyError):
    """A plugin package is malformed."""


class DuplicatePluginError(PluginError):
    pass


class InvalidPluginError(PluginError):
    pass


class MissingPluginsError(PluginError):
    """A plugin module does not define ``PLUGINS`` (catches a forgotten registration)."""


def choice[T: str](value: str, known: tuple[T, ...], option: str) -> T:
    """``value`` as one of ``known``, else a usage error naming ``option``."""
    for candidate in known:
        if value == candidate:
            return candidate
    raise UsageError(f"{option} must be one of {list(known)}, got {value!r}")
