"""Migrating a stored dataset to ``velocity_convention`` without bumping its version (WP-G 1.C).

APPLY is reopened for every simulated clip and run again on the whole plan; ``discard_downstream``
drops a render only where the poses, sizes or hidden frames changed. The manifest
``<dataset>/migrations/velocity-<convention>.json`` is written before anything changes and
completed afterwards, so a crashed run resumes and still lists every discarded render; once
completed, running it again changes nothing."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physverify.core.errors import ConfigError, MissingInputError
from physverify.synth.generate import load_plan
from physverify.synth.gt_export import DATASET_NAME
from physverify.synth.perturb import apply_variants
from physverify.synth.plan import DatasetPlan
from physverify.synth.settings import DatasetSettings
from physverify.synth.store import (
    RECORD,
    SKIPPED_FILE,
    DatasetStore,
    Step,
    atomic_write_json,
    read_json,
)

MIGRATIONS_DIR = "migrations"
CONVENTION_KEY = "velocity_convention"
RERENDER_KEY = "rerender_clip_ids"

type Clock = Callable[[], dt.datetime]


@dataclass(frozen=True)
class MigrationReport:
    manifest: Path
    applied: int
    rerender: tuple[str, ...]
    status_changes: Mapping[str, Any]


def manifest_path(store: DatasetStore, ds: DatasetSettings) -> Path:
    return store.root / MIGRATIONS_DIR / f"velocity-{ds.velocity_convention}.json"


def rerender_clip_ids(manifest: Path) -> list[str]:
    """The clips whose render a migration discarded (read by ``extract purge``)."""
    return [str(c) for c in read_json(manifest)[RERENDER_KEY]]


def manifest_dataset(manifest: Path) -> str:
    return str(read_json(manifest)["dataset"])


def clip_status(store: DatasetStore, clip_id: str) -> dict[str, Any]:
    if store.is_skipped(clip_id):
        return {"state": "skipped", "reason": store.skip_reason(clip_id)}
    state = "applied" if store.is_done(Step.APPLY, clip_id) else "pending"
    return {"state": state}


def statuses(plan: DatasetPlan, store: DatasetStore) -> dict[str, dict[str, Any]]:
    return {v.clip_id: clip_status(store, v.clip_id) for v in plan.variants}


def status_changes(
    before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Clips whose state (applied, skipped, pending) differs, with both sides."""
    return {
        clip_id: {"before": before.get(clip_id), "after": status}
        for clip_id, status in sorted(after.items())
        if (before.get(clip_id) or {}).get("state") != status["state"]
    }


def check_plan_hash(plan: DatasetPlan, ds: DatasetSettings) -> None:
    """The stored plan must be this content, either already migrated or from before the key."""
    known = {ds.content_hash(), ds.content_hash(without=(CONVENTION_KEY,))}
    if plan.config_hash not in known:
        raise ConfigError(
            f"dataset {plan.version} was planned with other settings than the current ones "
            "(beyond velocity_convention): migration refused"
        )


def baseline(
    plan: DatasetPlan, store: DatasetStore, ds: DatasetSettings, commit: str, now: Clock
) -> dict[str, Any]:
    clip_ids = [v.clip_id for v in plan.variants]
    return {
        "dataset": DATASET_NAME,
        "version": plan.version,
        "convention": ds.velocity_convention,
        "commit": commit,
        "created_utc": _stamp(now),
        "config_hash": {"old": plan.config_hash, "new": ds.content_hash()},
        "rendered_before": [c for c in clip_ids if store.is_done(Step.RENDER, c)],
        "status_before": statuses(plan, store),
        RERENDER_KEY: [],
    }


def load_or_start(
    path: Path, plan: DatasetPlan, store: DatasetStore, ds: DatasetSettings, commit: str, now: Clock
) -> dict[str, Any]:
    if not path.is_file():
        started = baseline(plan, store, ds, commit, now)
        atomic_write_json(path, started)
        return started
    stored: dict[str, Any] = read_json(path)
    if stored["config_hash"]["new"] != ds.content_hash():
        raise ConfigError(
            f"{path} migrated to config hash {stored['config_hash']['new']}, the current "
            f"settings hash to {ds.content_hash()}: remove the manifest only if that is intended"
        )
    return stored


def rewrite_plan_hash(store: DatasetStore, plan: DatasetPlan, new_hash: str) -> DatasetPlan:
    data = plan.to_json() | {"config_hash": new_hash}
    atomic_write_json(store.plan_path, data)
    return DatasetPlan.from_json(data)


def reopen_apply(plan: DatasetPlan, store: DatasetStore) -> int:
    """APPLY pending again for every simulated clip (skipped clips get a new chance)."""
    reopened = 0
    for variant in plan.variants:
        if not store.is_done(Step.SIMULATE, variant.sim_id):
            continue
        clip_dir = store.clip_dir(variant.clip_id)
        for name in (RECORD, SKIPPED_FILE):
            (clip_dir / name).unlink(missing_ok=True)
        reopened += 1
    return reopened


def migrate_velocity(
    store: DatasetStore,
    plugins: Mapping[str, Any],
    ds: DatasetSettings,
    commit: str,
    now: Clock = lambda: dt.datetime.now(dt.UTC),
) -> MigrationReport:
    """Re-run APPLY on every simulated clip of the stored plan under ``ds.velocity_convention``."""
    plan = load_plan(store)
    if plan is None:
        raise MissingInputError(f"nothing planned in {store.root}")
    check_plan_hash(plan, ds)
    path = manifest_path(store, ds)
    manifest = load_or_start(path, plan, store, ds, commit, now)
    if "completed_utc" in manifest:
        return _report(path, 0, manifest)
    plan = rewrite_plan_hash(store, plan, ds.content_hash())
    reopen_apply(plan, store)
    report = apply_variants(plan, store, plugins, ds)
    rerender = sorted(
        set(manifest[RERENDER_KEY])
        | {c for c in manifest["rendered_before"] if not store.is_done(Step.RENDER, c)}
    )
    after = statuses(plan, store)
    manifest |= {
        RERENDER_KEY: rerender,
        "status_after": after,
        "status_changes": status_changes(manifest["status_before"], after),
        "completed_utc": _stamp(now),
    }
    atomic_write_json(path, manifest)
    return _report(path, len(report.applied), manifest)


def _report(path: Path, applied: int, manifest: Mapping[str, Any]) -> MigrationReport:
    return MigrationReport(path, applied, tuple(manifest[RERENDER_KEY]), manifest["status_changes"])


def _stamp(now: Clock) -> str:
    return now().astimezone(dt.UTC).isoformat(timespec="seconds")
