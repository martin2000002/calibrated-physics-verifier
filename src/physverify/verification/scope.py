"""Which checks a consumer leaves out (H2 agreement, H3 scorer): one place for the ids.

The ids are validated by the consumer against its check registry (``checked``), so a typo fails
loudly instead of excluding nothing. The ids decided by the preregistered rules (ADR 0015) come
from the decision record ``exclusions.json`` and are merged here, so no consumer reads it."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

from pydantic import PrivateAttr

from physverify.calibration.exclusion_record import ExclusionRecord, load_record
from physverify.calibration.settings import active_checks_hash
from physverify.config.models import FrozenModel, Settings
from physverify.core.errors import ConfigError, UsageError

SECTION = "check_scope"


class CheckScope(FrozenModel):
    exclude_checks: tuple[str, ...] = ()
    """Checks no consumer decides with: the record's ids plus any manual ones."""
    time_dependent_checks: tuple[str, ...] = ("R5b", "R4_jump")
    """Checks whose magnitude depends on the clip's time base (fps, ADR 0006 R9)."""
    decision_record: str | None = None
    """The record of ``exclusions decide``, relative to ``paths.calibration_dir``."""
    require_decision_record: bool = False
    manual_deviation: str | None = None
    """Why manual ``exclude_checks`` are added to a record (a declared deviation)."""
    _record: ExclusionRecord | None = PrivateAttr(default=None)

    @property
    def record(self) -> ExclusionRecord | None:
        return self._record

    @property
    def record_sha256(self) -> str | None:
        return None if self._record is None else self._record.sha256

    def excluded(self, *, time_dependent: bool) -> frozenset[str]:
        """``exclude_checks``, plus the time-dependent ones when the consumer excludes them."""
        extra = self.time_dependent_checks if time_dependent else ()
        return frozenset((*self.exclude_checks, *extra))

    def checked(self, known: Collection[str]) -> CheckScope:
        unknown = sorted(set(self.exclude_checks + self.time_dependent_checks) - set(known))
        if unknown:
            raise ConfigError(f"[{SECTION}] names unknown checks {unknown}; known: {sorted(known)}")
        return self

    def with_record(self, record: ExclusionRecord) -> CheckScope:
        """This scope with the record's ids merged in; manual ids need a declared deviation."""
        if self.exclude_checks and not self.manual_deviation:
            raise ConfigError(
                f"[{SECTION}] has manual exclude_checks {list(self.exclude_checks)} next to the "
                "decision record: declare why in manual_deviation, or remove them"
            )
        merged = tuple(sorted({*self.exclude_checks, *record.exclude_checks}))
        scope = self.model_copy(update={"exclude_checks": merged})
        scope._record = record
        return scope


def record_path(scope: CheckScope, calibration_dir: Path) -> Path | None:
    return None if scope.decision_record is None else calibration_dir / scope.decision_record


def check_scope(settings: Settings, known: Collection[str]) -> CheckScope:
    """``[check_scope]`` with the decision record merged, its ids checked against ``known``."""
    scope = settings.section(SECTION, CheckScope)
    path = record_path(scope, settings.paths.calibration_dir)
    if path is None or not path.is_file():
        if scope.require_decision_record:
            raise ConfigError(
                f"[{SECTION}] requires a decision record and there is none at {path}: "
                "run `physverify exclusions decide`"
            )
        return scope.checked(known)
    try:
        record = load_record(path, active_checks_hash(settings))
    except UsageError as exc:
        raise ConfigError(str(exc)) from exc
    return scope.with_record(record).checked(known)


def require_record_for(scope: CheckScope, *, calibrated: bool, artifact_sha256: str | None) -> None:
    """A calibrated rule is frozen only on a decision record of the same artifact."""
    if not calibrated:
        return
    record = scope.record
    if record is None:
        raise UsageError(
            "a calibrated rule needs the exclusion decision record: run "
            "`physverify exclusions decide` first"
        )
    if record.inputs.artifact_sha256 != artifact_sha256:
        raise UsageError(
            f"the decision record was decided on artifact {record.inputs.artifact_sha256[:12]}, "
            f"not the consumed {str(artifact_sha256)[:12]}: decide again"
        )
