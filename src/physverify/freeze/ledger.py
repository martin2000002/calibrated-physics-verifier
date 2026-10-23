"""The sealed ledger's looks, the one retry and the v1 arm."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
from physverify.freeze.errors import QueryError


def look_state(entries: Sequence[LedgerEntry], specs: Sequence[str], *, retry: bool) -> str:
    """J6: a set is looked at once; one retry only when that look wrote no evaluation."""
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


def v1_state(
    entries: Sequence[LedgerEntry], frozen: Mapping[str, Any], specs: Sequence[str]
) -> str:
    """The v1 arm repeats a look of the freeze, once."""
    found = earlier(entries, specs)
    looks = [e for e in found if e.commit == frozen["commit"]]
    arms = [e for e in found if e.commit == frozen["v1"]["commit"]]
    if not looks:
        raise QueryError(f"{','.join(specs)} has no look of the freeze yet: run look first")
    if arms:
        raise QueryError(f"the v1 arm already looked at {','.join(specs)}: " + _lines(arms))
    return "v1arm"


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
