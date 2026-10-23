"""Pre-registration of the test comparison (ADR 0011 D10, Juicio changes 2 and 8).

Written by ``agreement freeze`` from the dev run, before any test run is compared. ``compare``
on the test subset recomputes every frozen hash and refuses to run when one changed. A refreeze
archives the file it replaces; once a test comparison exists it needs a declared deviation and
makes the H2 verdict post hoc. The chain and audit helpers here are shared with H3.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol

from pydantic import BaseModel, Field, ValidationError

from physverify.core.atomic import write_text_atomic
from physverify.core.digests import sha256_if_file
from physverify.core.errors import ConfigError, MissingInputError, UsageError
from physverify.core.models import FrozenModel

FROZEN_FIELDS = (
    "commit",
    "agreement_config_sha256",
    "subset_sha256",
    "queries_sha256",
    "calibration_artifact_sha256",
    "rule",
    "rule_label",
    "profile",
    "calibration_gate",
    "dev_icc",
    "dev_design_effect",
    "exclusion_record_sha256",
    "secondary_tests",
)
REPLACED_DIR: Final = "replaced"
COMPARISON_LOG_SUFFIX: Final = ".test_comparisons.jsonl"


class Preregistration(FrozenModel):
    created_utc: str
    commit: str
    agreement_config_sha256: str
    subset_sha256: dict[str, str | None]
    queries_sha256: str | None
    calibration_artifact_sha256: str | None
    rule: str
    rule_label: str
    profile: str
    device: str
    dev_run: str
    dev_subset: str
    test_subset: str
    n_dev: int
    dev_majority: int | None
    dev_cut: float | None
    dev_tuned_cut: float | None
    kubric_threshold: float | None
    calibration_gate: str | None = None
    """Why the calibrated stratum was not usable at freeze (the provisional rule ran)."""
    dev_icc: float | None = None
    dev_design_effect: float | None = None
    """Estimated on h2_dev (caption ANOVA ICC of the primary discordance); the clustered MDE of
    the test comparison uses it, so the test data do not decide how it is read."""
    exclusion_record_sha256: str | None = None
    """The ADR 0015 decision record the excluded checks came from (None: none was decided)."""
    secondary_tests: list[str] = Field(default_factory=list)
    replaces_sha256: str | None = None
    deviations: list[str] = Field(default_factory=list)
    post_hoc: bool = False


class FreezeAudit(FrozenModel):
    """How a report is bound to its pre-registration (H2 and H3). ``blind`` compares the first
    freeze with the earliest evaluated file time and ``frozen_after_eval`` the current freeze with
    it; both are informational only (file times can be touched)."""

    prereg_sha256: str
    versions: int
    created_utc: str
    replaces_sha256: str | None
    first_created_utc: str
    earliest_eval_utc: str | None
    blind: bool | None
    frozen_after_eval: bool | None = None
    post_hoc: bool
    frozen_commit: str
    report_commit: str
    deviations: list[str]


class LoggedComparison(FrozenModel):
    """One pre-registered comparison on test, logged so a later refreeze cannot hide it."""

    prereg_sha256: str
    run: str
    compared_utc: str


class Chained(Protocol):
    @property
    def replaces_sha256(self) -> str | None: ...


def archive_replaced(path: Path, archive: Callable[[str], Path]) -> None:
    """Copies the file at ``path`` (if any) to ``archive(its sha256)`` before it is replaced."""
    sha = sha256_if_file(path)
    if sha is None:
        return
    target = archive(sha)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())


def walk_chain[T: Chained](record: T, load: Callable[[str], T | None]) -> list[T]:
    """``record`` and every version it replaces, newest first (stops at a missing link)."""
    found = [record]
    seen: set[str] = set()
    while (sha := found[-1].replaces_sha256) is not None and sha not in seen:
        seen.add(sha)
        older = load(sha)
        if older is None:
            break
        found.append(older)
    return found


def earliest_mtime(paths: Iterable[Path]) -> str | None:
    times = [p.stat().st_mtime for p in paths if p.is_file()]
    if not times:
        return None
    return dt.datetime.fromtimestamp(min(times), dt.UTC).isoformat(timespec="seconds")


def frozen_before(first_created: str, earliest: str | None) -> bool | None:
    if earliest is None:
        return None
    return dt.datetime.fromisoformat(first_created) <= dt.datetime.fromisoformat(earliest)


def frozen_after(created: str, earliest: str | None) -> bool | None:
    blind = frozen_before(created, earliest)
    return None if blind is None else not blind


def compared_under_other(logged: Sequence[LoggedComparison], sha256: str) -> bool:
    """A test comparison was made under another pre-registration than ``sha256``."""
    return any(entry.prereg_sha256 != sha256 for entry in logged)


def replaced_path(path: Path, sha256: str) -> Path:
    return path.parent / REPLACED_DIR / f"{sha256}.json"


def write_preregistration(path: Path, record: Preregistration) -> None:
    archive_replaced(path, lambda sha: replaced_path(path, sha))
    write_text_atomic(path, record.model_dump_json(indent=2) + "\n")


def load_preregistration(path: Path) -> Preregistration:
    if not path.is_file():
        raise MissingInputError(
            f"{path} is missing; run `physverify agreement freeze` on the dev run"
        )
    try:
        return Preregistration.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ConfigError(f"{path} is invalid:\n{exc}") from exc


def preregistration_chain(path: Path, record: Preregistration) -> list[Preregistration]:
    def load(sha: str) -> Preregistration | None:
        archived = replaced_path(path, sha)
        return load_preregistration(archived) if archived.is_file() else None

    return walk_chain(record, load)


def comparison_log_path(path: Path) -> Path:
    return path.with_name(path.stem + COMPARISON_LOG_SUFFIX)


def logged_comparisons(path: Path) -> list[LoggedComparison]:
    log = comparison_log_path(path)
    if not log.is_file():
        return []
    lines = log.read_text(encoding="utf-8").splitlines()
    return [LoggedComparison.model_validate_json(line) for line in lines if line.strip()]


def log_comparison(path: Path, entry: LoggedComparison) -> None:
    log = comparison_log_path(path)
    previous = log.read_text(encoding="utf-8") if log.is_file() else ""
    write_text_atomic(log, previous + entry.model_dump_json() + "\n")


def plan_refreeze(
    old: Preregistration | None,
    old_sha256: str | None,
    current: Preregistration,
    *,
    force: bool,
    deviation: str | None,
    tested: bool,
) -> Preregistration | None:
    """The record to write, or None to keep the old one. ``tested``: a pre-registered test
    comparison exists, so any (re)freeze is post hoc and must declare why."""
    changed = [] if old is None else changed_fields(old, current.model_dump())
    if old is not None and not changed and not force:
        return None
    if old is not None and not force:
        raise UsageError(
            f"the frozen state changed ({', '.join(changed)}); "
            "the test comparison is already pre-registered (refreeze with --force)"
        )
    if tested and deviation is None:
        raise UsageError(
            "a pre-registered test comparison already exists; a refreeze is post hoc: "
            "add --declare-deviation TEXT"
        )
    added = [] if deviation is None else [f"{deviation} (changed: {changed})"]
    return current.model_copy(
        update={
            "replaces_sha256": old_sha256,
            "deviations": [*([] if old is None else old.deviations), *added],
            "post_hoc": tested or (old is not None and old.post_hoc),
        }
    )


def changed_fields(
    frozen: BaseModel, current: Mapping[str, object], *, fields: Sequence[str] = FROZEN_FIELDS
) -> list[str]:
    """Names of the frozen fields whose current value differs."""
    stored = frozen.model_dump()
    return [name for name in fields if name in current and stored[name] != current[name]]


def require_unchanged(
    frozen: BaseModel, current: Mapping[str, object], *, fields: Sequence[str] = FROZEN_FIELDS
) -> None:
    changed = changed_fields(frozen, current, fields=fields)
    if changed:
        stored = frozen.model_dump()
        detail = ", ".join(f"{n}: {stored[n]!r} -> {current[n]!r}" for n in changed)
        raise ConfigError(
            f"the pre-registration no longer holds ({detail}); the test is not comparable"
        )
