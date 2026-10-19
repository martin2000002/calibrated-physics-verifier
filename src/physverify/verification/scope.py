"""Which checks a consumer leaves out (H2 agreement, H3 scorer): one place for the ids.

Pure: the ids are validated by the consumer against its check registry (``checked``), so a
typo fails loudly instead of excluding nothing."""

from __future__ import annotations

from collections.abc import Collection

from physverify.config.models import FrozenModel, Settings
from physverify.core.errors import ConfigError

SECTION = "check_scope"


class CheckScope(FrozenModel):
    exclude_checks: tuple[str, ...] = ()
    """Checks no consumer decides with (decided by the author, items 1.1/1.5)."""
    time_dependent_checks: tuple[str, ...] = ("R5b", "R4_jump")
    """Checks whose magnitude depends on the clip's time base (fps, ADR 0006 R9)."""

    def excluded(self, *, time_dependent: bool) -> frozenset[str]:
        """``exclude_checks``, plus the time-dependent ones when the consumer excludes them."""
        extra = self.time_dependent_checks if time_dependent else ()
        return frozenset((*self.exclude_checks, *extra))

    def checked(self, known: Collection[str]) -> CheckScope:
        unknown = sorted(set(self.exclude_checks + self.time_dependent_checks) - set(known))
        if unknown:
            raise ConfigError(f"[{SECTION}] names unknown checks {unknown}; known: {sorted(known)}")
        return self


def check_scope(settings: Settings, known: Collection[str]) -> CheckScope:
    """The ``[check_scope]`` section, with its ids checked against ``known`` check ids."""
    return settings.section(SECTION, CheckScope).checked(known)
