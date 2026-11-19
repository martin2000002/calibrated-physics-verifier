"""The sealed ledger's looks, the one retry and the v1 arm. A look is ``spec@version`` for a
dataset whose content has versions (``calibration.ledger_versions``)."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from physverify.calibration.ledger_versions import (
    LegacySettings,
    legacy_settings,
    looks_of,
    resolved,
)
from physverify.calibration.report import EVALUATION_FILE, EVALUATION_MD
from physverify.calibration.sealed_ledger import (
    LedgerEntry,
    earlier,
    ledger_path,
    read_entries,
    sealed_settings,
)
from physverify.config.models import Settings
from physverify.core.errors import PhysverifyError
from physverify.datasets._base import DatasetSource, dataset_from
from physverify.freeze.errors import QueryError


def look_state(entries: Sequence[LedgerEntry], specs: Sequence[str], *, retry: bool) -> str:
    """A set is looked at once; one retry only when that look wrote no evaluation."""
    found = earlier(entries, specs)
    if not retry:
        if found:
            raise QueryError(
                f"{','.join(specs)} already has {len(found)} ledger lines: " + _lines(found)
            )
        return "first"
    if len(found) != 1:
        raise QueryError(f"--retry needs exactly one earlier look, found {len(found)}")
    run_dir = found[0].run_dir
    if run_dir is None:
        raise QueryError("the earlier look recorded no run directory")
    written = [name for name in (EVALUATION_FILE, EVALUATION_MD) if (Path(run_dir) / name).exists()]
    if written:
        raise QueryError(f"the earlier look wrote {written} in {run_dir}: no retry")
    return "retry"


def recorded_look(
    entries: Sequence[LedgerEntry], run_dir: Path, commit: str, specs: Sequence[str]
) -> str:
    """The look of ``run_dir`` left its ledger line, at ``commit`` and over exactly ``specs``: a
    look the guard exempted would otherwise pass as never made."""
    for entry in entries:
        if (
            entry.run_dir is not None
            and Path(entry.run_dir).resolve() == run_dir.resolve()
            and entry.commit == commit
            and sorted(entry.sets) == sorted(specs)
        ):
            return f"recorded: {entry.utc} {entry.command} freeze {entry.freeze[:12]}"
    raise QueryError(
        f"no ledger line of run {run_dir} at commit {commit[:12]} covers exactly {','.join(specs)}"
    )


def v1_state(
    entries: Sequence[LedgerEntry],
    frozen: Mapping[str, Any],
    looks: Mapping[str, str],
    legacy: LegacySettings,
) -> str:
    """The v1 arm repeats a look of the freeze, once. The v1 commit's code records bare specs, so
    its lines are found by spec and every other line by look."""
    specs = list(looks)
    v1 = frozen["v1"]["commit"]
    arms = [e for e in earlier(entries, specs) if e.commit == v1]
    ours = resolved_entries([e for e in entries if e.commit != v1], looks, legacy)
    made = [e for e in earlier(ours, list(looks.values())) if e.commit == frozen["commit"]]
    if not made:
        raise QueryError(f"{','.join(specs)} has no look of the freeze yet: run look first")
    if arms:
        raise QueryError(f"the v1 arm already looked at {','.join(specs)}: " + _lines(arms))
    return "v1arm"


def resolved_entries(
    entries: Sequence[LedgerEntry], looks: Mapping[str, str], legacy: LegacySettings
) -> tuple[LedgerEntry, ...]:
    try:
        return resolved(entries, looks, legacy)
    except PhysverifyError as exc:
        raise QueryError(str(exc)) from exc


def current_looks(
    settings: Settings, registry: Mapping[str, type[DatasetSource]], specs: Sequence[str]
) -> dict[str, str]:
    """Each spec -> the look the guard records for it under ``settings``."""
    try:
        source_of = functools.partial(dataset_from, registry, settings=settings)
        return looks_of(specs, source_of, settings.sections)
    except PhysverifyError as exc:
        raise QueryError(str(exc)) from exc


def ledger_looks(
    settings: Settings, home: Path, looks: Mapping[str, str]
) -> tuple[LedgerEntry, ...]:
    """The ledger with its legacy bare lines named by the version they looked at."""
    return resolved_entries(ledger_entries(settings, home), looks, legacy_settings(settings))


def _lines(entries: Sequence[LedgerEntry]) -> str:
    return "; ".join(f"{e.utc} {e.commit[:12]} {e.command} run {e.run_dir}" for e in entries)


def ledger_file(settings: Settings, home: Path) -> Path:
    """The ledger the guard of every look appends to."""
    return ledger_path(sealed_settings(settings), home, settings.repo_root)


def ledger_entries(settings: Settings, home: Path) -> tuple[LedgerEntry, ...]:
    """Fails closed when the ledger is missing, as the guard does."""
    try:
        return read_entries(ledger_file(settings, home))
    except PhysverifyError as exc:
        raise QueryError(str(exc)) from exc


def ledger_line(settings: Settings, home: Path) -> str:
    entries = ledger_entries(settings, home)
    return f"ledger {ledger_file(settings, home)}: {len(entries)} lines"
