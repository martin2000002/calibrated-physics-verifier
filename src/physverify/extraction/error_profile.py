"""The measured per-stage error profile the providers read, when one has been written."""

from __future__ import annotations

from pathlib import Path

from physverify.core.errors import ConfigError
from physverify.extraction.cache import plan_entry_filter, profile_keys
from physverify.extraction.plan import ResolvedPlan
from physverify.schema.error_profile import ErrorEntry, ErrorProfile
from physverify.schema.stages import Stage


def load_error_profile(path: Path) -> ErrorProfile | None:
    return ErrorProfile.load_json(path) if path.is_file() else None


def plan_profile_keys(plan: ResolvedPlan) -> frozenset[tuple[Stage, str]]:
    """The (stage, provider) profile entries some provider of ``plan`` reads."""
    names = dict(plan.names)
    return frozenset(key for step in plan.stages for key in profile_keys(step.provider, names))


def stale_entries(plan: ResolvedPlan, profile: ErrorProfile) -> tuple[ErrorEntry, ...]:
    """Entries ``plan`` reads that were measured under another profile or stage params: ignored,
    so their sigma is unknown."""
    return profile.rejected(plan_profile_keys(plan), plan_entry_filter(plan))


def require_current_profile(plan: ResolvedPlan, profile: ErrorProfile | None) -> None:
    """Refuse a plan whose providers read a missing or stale profile (``--strict-profile``)."""
    rerun = "re-run `diagnose stages --write-profile` for this profile"
    if profile is None:
        if plan_profile_keys(plan):
            raise ConfigError(f"--strict-profile: no error profile is loaded; {rerun}")
        return
    stale = sorted({f"{e.stage.value}/{e.provider}" for e in stale_entries(plan, profile)})
    if stale:
        raise ConfigError(
            f"--strict-profile: the error profile is stale for {', '.join(stale)} "
            f"(measured under another profile or stage params); {rerun}"
        )
