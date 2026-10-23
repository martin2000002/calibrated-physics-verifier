"""The sealed ledger (ADR 0015 amendment 2 section 5): the one-look sets of ``[sealed]`` and the
line a look leaves before any number is computed. Pure except ``read_entries``/``append_entry``.

A look is a frozen state, not a command: every line carries ``freeze`` (commit of a clean tree,
the hash of the effective sections, profile and device, and the sha256 of every artifact a
verdict reads), so the final evaluation may run several commands on one set, while a look under
another frozen state needs an explicit repeat. The ledger is created once by hand and never by
``physverify``, so a wrong HOME or path fails closed instead of starting an empty ledger."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError, field_validator

from physverify.config.models import Settings, _anchor
from physverify.core.errors import ConfigError, PhysverifyError, UsageError
from physverify.core.models import FrozenModel
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.io import stable_hash

SECTION = "sealed"
UNSEAL_ENV = "PHYSVERIFY_UNSEAL"
"""Comma-separated specs, spelled as in ``[sealed] sets``, this invocation may look at."""
REPEAT_ENV = "PHYSVERIFY_UNSEAL_REPEAT"
SPEC_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(:(calib|eval))?$")


class SealedSetError(UsageError):
    """A sealed set would be looked at without the explicit unseal (or repeat) variable."""


class LedgerError(PhysverifyError):
    """The ledger cannot be read or written: the command stops before computing."""


class SealedSettings(FrozenModel):
    ledger: Path
    sets: tuple[str, ...]

    @field_validator("sets")
    @classmethod
    def _specs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        bad = [spec for spec in value if not SPEC_PATTERN.match(spec)]
        if bad:
            raise ValueError(f"specs must be dataset or dataset:calib|eval, got {bad}")
        return value


def sealed_settings(settings: Settings) -> SealedSettings:
    """Fails closed: without ``[sealed]`` nothing would be guarded."""
    if SECTION not in settings.sections:
        raise ConfigError(
            f"no [{SECTION}] section (configs/sealed.toml): the sealed sets are unknown"
        )
    return settings.section(SECTION, SealedSettings)


def ledger_path(sealed: SealedSettings, home: Path, root: Path) -> Path:
    """``~`` is ``home`` (the invoking HOME), a relative path is under ``root``."""
    return _anchor(sealed.ledger, root, home)


@dataclass(frozen=True)
class SealedSpec:
    dataset: str
    split: Split | None

    @classmethod
    def parse(cls, text: str) -> SealedSpec:
        dataset, _, split = text.partition(":")
        return cls(dataset, Split(split) if split else None)

    def covers(self, dataset: str, clips: Sequence[ClipManifest]) -> bool:
        if dataset != self.dataset:
            return False
        return any(self.split is None or clip.split is self.split for clip in clips)


def touched(dataset: str, clips: Sequence[ClipManifest], specs: Sequence[str]) -> tuple[str, ...]:
    """The specs a selection of ``dataset`` clips looks at."""
    return tuple(spec for spec in specs if SealedSpec.parse(spec).covers(dataset, clips))


def unsealed(value: str | None) -> frozenset[str]:
    return frozenset(part.strip() for part in (value or "").split(",") if part.strip())


def require_unsealed(found: Sequence[str], allowed: Collection[str]) -> None:
    missing = [spec for spec in found if spec not in allowed]
    if missing:
        raise SealedSetError(
            f"sealed sets {missing} are one-look sets (ADR 0015 amendment 2 section 5): set "
            f"{UNSEAL_ENV}={','.join(found)} to spend the look; nothing was computed"
        )


def look_config_hash(settings: Settings) -> str:
    """The configuration a verdict depends on: sections, profile and device. Paths, threads and
    the lock are left out so the commands of one look may differ in them; what the paths point
    to enters the freeze as file sha256."""
    return stable_hash(
        {
            "profile": settings.profile,
            "device": settings.runtime.device,
            "sections": dict(settings.sections),
        }
    )


def freeze_of(commit: str, config_hash: str, artifacts: Mapping[str, str | None]) -> str:
    """The frozen state of a look: the same in every command run on it."""
    return stable_hash({"commit": commit, "config_hash": config_hash, "artifacts": artifacts})


def require_clean(commit: str, dirty: str | None) -> None:
    """``dirty``: ``git status --porcelain`` of the tree the commit names (None: no checkout)."""
    if dirty is None or not commit:
        raise SealedSetError("a sealed look needs a git checkout: the commit names the freeze")
    if dirty.strip():
        raise SealedSetError(
            "uncommitted or untracked files under src or configs: commit them before a sealed "
            f"look, or its freeze would not be the commit's:\n{dirty.rstrip()}"
        )


class LedgerDataset(FrozenModel):
    name: str
    subset: str | None
    n_clips: int
    clip_ids_hash: str


class LedgerEntry(FrozenModel):
    utc: str
    commit: str
    command: str
    argv: tuple[str, ...]
    sets: tuple[str, ...]
    """The sealed specs this look touched."""
    sealed: tuple[str, ...]
    """Every spec ``[sealed] sets`` held at the time."""
    datasets: tuple[LedgerDataset, ...]
    artifacts: dict[str, str]
    config_hash: str
    """``look_config_hash``: sections, profile and device (not paths, threads or the lock)."""
    freeze: str
    repeat: bool
    run_dir: str | None
    user: str
    host: str


def dataset_entry(name: str, subset: str | None, clips: Sequence[ClipManifest]) -> LedgerDataset:
    ids = sorted(clip.clip_id for clip in clips)
    return LedgerDataset(name=name, subset=subset, n_clips=len(ids), clip_ids_hash=stable_hash(ids))


def other_freezes(entries: Sequence[LedgerEntry], found: Collection[str], freeze: str) -> list[str]:
    """The touched specs some earlier line looked at under another frozen state."""
    return sorted({s for e in entries if e.freeze != freeze for s in e.sets if s in found})


def check_repeat(
    entries: Sequence[LedgerEntry], found: Collection[str], freeze: str, allow: bool
) -> bool:
    """Whether the line is a repeat: a look under another frozen state (only with ``allow``), or
    any look with ``allow`` at a set some line already covers (run_final.sh's declared retry)."""
    repeated = other_freezes(entries, found, freeze)
    if repeated and not allow:
        raise SealedSetError(
            f"sealed sets {repeated} were already looked at under another frozen state "
            f"(commit, config or artifacts changed): set {REPEAT_ENV}=1 to record a repeat look"
        )
    return bool(repeated) or (allow and bool(earlier(entries, found)))


def earlier(entries: Sequence[LedgerEntry], found: Collection[str]) -> list[LedgerEntry]:
    return [e for e in entries if any(s in found for s in e.sets)]


def require_ledger(path: Path) -> None:
    if not path.is_file():
        raise LedgerError(
            f"no sealed ledger at {path}: it is created once by hand (touch {path}), never by a "
            "look; check HOME (the H200 scripts source env.sh) and [sealed] ledger"
        )


def read_entries(path: Path) -> tuple[LedgerEntry, ...]:
    """Every line of the ledger; a missing, unreadable or malformed one fails closed."""
    require_ledger(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return tuple(LedgerEntry.model_validate_json(line) for line in lines if line.strip())
    except (OSError, ValidationError) as exc:
        raise LedgerError(f"cannot read the sealed ledger {path}: {exc}") from exc


def append_entry(path: Path, entry: LedgerEntry) -> None:
    """One JSON line with sorted keys, flushed to disk before the command computes anything."""
    line = json.dumps(entry.model_dump(mode="json"), sort_keys=True)
    require_ledger(path)
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise LedgerError(f"cannot write the sealed ledger {path}: {exc}") from exc
