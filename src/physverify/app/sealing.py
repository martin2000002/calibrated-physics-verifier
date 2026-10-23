"""The sealed-ledger guard (``calibration.sealed_ledger``) of the commands that compute numbers on
dataset clips, and the table that classifies every subcommand: a new one fails a test until it is
listed here, so no command escapes the guard by omission."""

from __future__ import annotations

import datetime as dt
import socket
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from physverify.app.context import AppContext
from physverify.calibration.sealed_ledger import (
    REPEAT_ENV,
    UNSEAL_ENV,
    LedgerEntry,
    append_entry,
    check_repeat,
    dataset_entry,
    earlier,
    freeze_of,
    frozen_files,
    ledger_path,
    locked,
    look_config_hash,
    read_entries,
    require_clean,
    require_unsealed,
    sealed_settings,
    touched,
    unsealed,
)
from physverify.core.digests import sha256_if_file
from physverify.core.output import Echo
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
"""Run the checks or measure stage errors on the clips of a dataset. These are the labels the
ledger records, persisted in sealed ledgers: a renamed leaf keeps its label."""

LEDGER_LABELS_BY_LEAF: Final[Mapping[str, str]] = MappingProxyType(
    {
        "calibrate dev-score": "calibrate devscore",
        "selective-risk curve": "abstain curve",
        "selective-risk fit": "abstain fit",
    }
)
"""CLI leaf -> the ledger label it writes, where the two differ."""


def ledger_label(leaf: str) -> str:
    """The label the ledger records for CLI leaf ``leaf``."""
    return LEDGER_LABELS_BY_LEAF.get(leaf, leaf)


_LEAVES_BY_LABEL: Final = {label: leaf for leaf, label in LEDGER_LABELS_BY_LEAF.items()}
GUARDED_LEAVES: Final = frozenset(_LEAVES_BY_LABEL.get(label, label) for label in GUARDED)
"""The CLI leaves whose label is in ``GUARDED``."""

_DATA = "builds, fetches or lists data; computes no check or error number"
_READS = "re-reads files a guarded command wrote (its look is already in the ledger)"
_GENERATED = (
    "scores generated videos, not a dataset split "
    "(the generator comparison has its own pre-registration)"
)
_TIMING = "measures wall time only"
EXEMPT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "agreement compare": "VideoPhy h2_test keeps its own ledger (run_human_agreement.sh)",
        "agreement fetch": _DATA,
        "agreement freeze": "reads the h2_dev verify run (development data)",
        "agreement subset": _DATA,
        "calibrate publish": _READS,
        "calibrate show": _READS,
        "compare-generators compare": _GENERATED,
        "compare-generators freeze": _GENERATED,
        "compare-generators gate-autoeval": _GENERATED,
        "compare-generators generate": _GENERATED,
        "compare-generators judge": _GENERATED,
        "compare-generators rescore": _GENERATED,
        "compare-generators score": _GENERATED,
        "compare-generators status": _GENERATED,
        "demo report": _GENERATED,
        "demo run": _GENERATED + "; a --video path is a file, not a dataset selection",
        "demo scenes": _GENERATED,
        "doctor": _DATA,
        "download fetch": _DATA,
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
        "results": _READS,
        "scene-stats delta": _READS,
        "scene-stats table": _READS,
        "size e2e-eval": "Objectron through sizing/, not a DatasetSource",
        "size e2e-prepare": _DATA,
        "size eval": "Objectron through sizing/, not a DatasetSource",
        "size fetch": _DATA,
        "size fit": "fits on Objectron train",
        "synth build": _DATA,
        "synth export-gt": _DATA,
        "synth generate": _DATA,
        "synth list": _DATA,
        "synth migrate": _DATA,
        "synth migrate-velocity": _DATA,
        "synth perturb": _DATA,
        "synth pull-image": _DATA,
        "synth render": _DATA,
        "synth simulate": _DATA,
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
    err: Echo,
    artifacts: Mapping[str, str] | None = None,
    run_dir: Path | None = None,
) -> None:
    """Nothing when no selection touches a sealed set; otherwise refuse without the unseal
    variable, or append the ledger line before returning. ``artifacts`` (label -> sha256) are
    the files the command judges with beyond the default ones; they enter the freeze. ``err``
    prints one line on the error stream."""
    if command not in GUARDED_LEAVES:
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
    err(f"ledger: {path}")
    look = _look(context, found, artifacts or {})
    with locked(path):
        previous = read_entries(path)
        repeat = check_repeat(previous, found, look.freeze, environ.get(REPEAT_ENV) == "1")
        for entry in earlier(previous, found):
            err(
                f"earlier look at {','.join(entry.sets)}: {entry.utc} commit {entry.commit[:12]} "
                f"{entry.command} freeze {entry.freeze[:12]}"
            )
        entry = look.model_copy(
            update={
                "command": ledger_label(command),
                "datasets": tuple(dataset_entry(s.dataset, s.subset, s.clips) for s in selections),
                "repeat": repeat,
                "run_dir": None if run_dir is None else str(run_dir),
            }
        )
        append_entry(path, entry)
    err(f"sealed look: {','.join(found)} freeze {look.freeze[:12]}")


def _tree_state(root: Path) -> str | None:
    """``git status --porcelain -- src configs`` (None outside a checkout)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "-uall", "--", "src", "configs"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _look(context: AppContext, found: tuple[str, ...], artifacts: Mapping[str, str]) -> LedgerEntry:
    """The invocation's part of the line: who, where, and the frozen state it looks with."""
    settings = context.settings
    commit = current_commit(settings.repo_root)
    require_clean(commit, _tree_state(settings.repo_root))
    config_hash = look_config_hash(settings)
    frozen = {label: sha256_if_file(path) for label, path in frozen_files(settings).items()}
    frozen |= dict(artifacts)
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
