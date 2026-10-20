"""The sealed-ledger guard (``calibration.sealed``) of the commands that compute numbers on dataset
clips, and the table that classifies every subcommand: a new one fails a test until it is listed
here, so no command escapes the guard by omission."""

from __future__ import annotations

import datetime as dt
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import typer

from physverify.calibration.sealed import (
    REPEAT_ENV,
    UNSEAL_ENV,
    LedgerEntry,
    append_entry,
    check_repeat,
    dataset_entry,
    earlier,
    file_sha,
    freeze_of,
    ledger_path,
    read_entries,
    require_unsealed,
    sealed_settings,
    touched,
    unsealed,
)
from physverify.cli.context import AppContext
from physverify.runtime.runs import current_commit
from physverify.schema.clip import ClipManifest

GUARDED: Final = frozenset(
    {
        "abstain curve",
        "abstain fit",
        "calibrate devscore",
        "calibrate evaluate",
        "calibrate fit",
        "calibrate stage-sigma",
        "diagnose attribution",
        "diagnose end-to-end",
        "diagnose real",
        "diagnose stages",
        "scene-stats per-clip",
        "verify",
    }
)
"""Run the checks or measure stage errors on the clips of a dataset."""

_DATA = "builds, fetches or lists data; computes no check or error number"
_READS = "re-reads files a guarded command wrote (its look is already in the ledger)"
_GENERATED = "scores generated videos, not a dataset split (H3 has its own pre-registration)"
_TIMING = "measures wall time only"
EXEMPT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "agreement compare": "VideoPhy h2_test keeps its own ledger (run_h2.sh, WP-N)",
        "agreement fetch": _DATA,
        "agreement freeze": "reads the h2_dev verify run (development data)",
        "agreement subset": _DATA,
        "calibrate publish": _READS,
        "calibrate show": _READS,
        "data fetch": _DATA,
        "dataset audit-migration": _DATA,
        "dataset build": _DATA,
        "dataset export-gt": _DATA,
        "dataset generate": _DATA,
        "dataset list": _DATA,
        "dataset migrate": _DATA,
        "dataset migrate-velocity": _DATA,
        "dataset perturb": _DATA,
        "dataset pull-image": _DATA,
        "dataset render": _DATA,
        "dataset simulate": _DATA,
        "demo report": _GENERATED,
        "demo run": _GENERATED,
        "demo scenes": _GENERATED,
        "doctor": _DATA,
        "exclusions decide": _READS,
        "exclusions r2-clips": _DATA,
        "exclusions show": _READS,
        "extract fetch-weights": _DATA,
        "extract purge": _DATA,
        "extract run": "fills the stage cache and prints no number (fresh controls need it)",
        "finetune beta-grid": _GENERATED,
        "finetune budget": _GENERATED,
        "finetune generate": _GENERATED,
        "finetune pairs": _GENERATED,
        "finetune pilot-review": _GENERATED,
        "finetune prompts": _GENERATED,
        "finetune score": _GENERATED,
        "finetune select": _GENERATED,
        "finetune status": _GENERATED,
        "finetune train": _GENERATED,
        "h3 compare": _GENERATED,
        "h3 freeze": _GENERATED,
        "h3 gate-autoeval": _GENERATED,
        "h3 generate": _GENERATED,
        "h3 judge": _GENERATED,
        "h3 rescore": _GENERATED,
        "h3 score": _GENERATED,
        "h3 status": _GENERATED,
        "results": _READS,
        "scene-stats delta": _READS,
        "scene-stats table": _READS,
        "size e2e-eval": "Objectron through sizing/, not a DatasetSource (WP-O Juicio 7)",
        "size e2e-prepare": _DATA,
        "size eval": "Objectron through sizing/, not a DatasetSource (WP-O Juicio 7)",
        "size fetch": _DATA,
        "size fit": "fits on Objectron train (WP-O Juicio 7)",
        "timing budget": _TIMING,
        "timing env": _TIMING,
        "timing fetch": _DATA,
        "timing list": _TIMING,
        "timing run": _TIMING,
        "timing summarize": _TIMING,
    }
)
"""Subcommand -> why it needs no guard."""


@dataclass(frozen=True)
class Selection:
    dataset: str
    subset: str | None
    clips: tuple[ClipManifest, ...]


def guard_sealed(
    context: AppContext,
    command: str,
    selections: Sequence[Selection],
    *,
    artifacts: Mapping[str, str] | None = None,
    run_dir: Path | None = None,
) -> None:
    """Nothing when no selection touches a sealed set; otherwise refuse without the unseal
    variable, or append the ledger line (``artifacts``: label -> sha256) before returning."""
    if command not in GUARDED:
        raise ValueError(f"{command!r} is not a guarded subcommand")
    sealed = sealed_settings(context.settings)
    found = tuple(
        dict.fromkeys(s for sel in selections for s in touched(sel.dataset, sel.clips, sealed.sets))
    )
    if not found:
        return
    environ = context.invocation.environ
    require_unsealed(found, unsealed(environ.get(UNSEAL_ENV)))
    home = Path(environ.get("HOME") or Path.home())
    path = ledger_path(sealed, home, context.settings.repo_root)
    typer.echo(f"ledger: {path}", err=True)
    look = _look(context, found)
    previous = read_entries(path)
    repeat = check_repeat(previous, found, look.freeze, environ.get(REPEAT_ENV) == "1")
    for entry in earlier(previous, found):
        typer.echo(
            f"earlier look at {','.join(entry.sets)}: {entry.utc} commit {entry.commit[:12]} "
            f"{entry.command} freeze {entry.freeze[:12]}",
            err=True,
        )
    entry = look.model_copy(
        update={
            "command": command,
            "datasets": tuple(dataset_entry(s.dataset, s.subset, s.clips) for s in selections),
            "artifacts": look.artifacts | dict(artifacts or {}),
            "repeat": repeat,
            "run_dir": None if run_dir is None else str(run_dir),
        }
    )
    append_entry(path, entry)
    typer.echo(f"sealed look: {','.join(found)} freeze {look.freeze[:12]}", err=True)


def _look(context: AppContext, found: tuple[str, ...]) -> LedgerEntry:
    """The invocation's part of the line: who, where, and the frozen state it looks with."""
    settings = context.settings
    commit, config_hash = current_commit(settings.repo_root), settings.config_hash()
    frozen = {
        "thresholds": file_sha(settings.paths.thresholds),
        "error_profile": file_sha(settings.paths.error_profile),
    }
    return LedgerEntry(
        utc=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        commit=commit,
        command="",
        argv=context.invocation.argv,
        sets=found,
        sealed=sealed_settings(settings).sets,
        datasets=(),
        artifacts={k: v for k, v in frozen.items() if v is not None},
        config_hash=config_hash,
        freeze=freeze_of(commit, config_hash, frozen),
        repeat=False,
        run_dir=None,
        user=context.invocation.environ.get("USER", ""),
        host=socket.gethostname(),
    )
