"""Migrating a stored dataset without bumping its version: to ``velocity_convention`` (WP-G 1.C,
``velocity-<convention>``) or through the ground-truth fixes of ``plan.GT_MIGRATIONS`` (WP-H
``floor-crossing``, ...), each run only as the first one the plan still lacks. Every migration
re-runs APPLY with this checkout, so it leaves the plan at ``GT_REVISION`` and its manifest lists
the fixes it ``covers``.

APPLY is reopened and run again on every clip it had already reached (applied or skipped; a clip
never applied stays pending for its own build); ``discard_downstream``
drops a render only where the poses, sizes or hidden frames changed, and a clip APPLY now refuses
loses everything it had (render and export included). The manifest
``<dataset>/migrations/<name>.json`` is written before anything changes and
completed afterwards, so a crashed run resumes and still lists every discarded render; once
completed, running it again changes nothing."""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physverify.core.errors import ConfigError, MissingInputError, UsageError
from physverify.synth.generate import load_plan
from physverify.synth.gt_export import DATASET_NAME, write_index
from physverify.synth.perturb import apply_variants
from physverify.synth.plan import GT_MIGRATIONS, GT_REVISION, DatasetPlan
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
COMPLETED_KEY = "completed_utc"
COVERS_KEY = "covers"
FLOOR_CROSSING = GT_MIGRATIONS[0]

type Clock = Callable[[], dt.datetime]


@dataclass(frozen=True)
class MigrationReport:
    manifest: Path
    applied: int
    rerender: tuple[str, ...]
    status_changes: Mapping[str, Any]


def velocity_migration(ds: DatasetSettings) -> str:
    return f"velocity-{ds.velocity_convention}"


def migration_names(ds: DatasetSettings) -> tuple[str, ...]:
    return (velocity_migration(ds), *GT_MIGRATIONS)


def check_name(name: str, ds: DatasetSettings) -> None:
    if name not in migration_names(ds):
        raise UsageError(f"unknown migration {name!r}; known: {list(migration_names(ds))}")


def check_pending(plan: DatasetPlan, name: str) -> None:
    """A ground-truth migration runs only as the first one the plan has not had."""
    pending = plan.pending_migration()
    if name in GT_MIGRATIONS and name != pending:
        state = "is up to date" if pending is None else f"needs `{pending}` first"
        raise ConfigError(
            f"dataset {plan.version} has ground-truth revision {plan.gt_revision} and {state}: "
            f"migration {name!r} refused"
        )


def covered(old_revision: int) -> list[str]:
    """The ground-truth fixes a run from ``old_revision`` brings in (APPLY uses this checkout)."""
    return list(GT_MIGRATIONS[old_revision:GT_REVISION])


def manifest_path(store: DatasetStore, name: str) -> Path:
    return store.root / MIGRATIONS_DIR / f"{name}.json"


def rerender_clip_ids(manifest: Path) -> list[str]:
    """The clips whose render a completed migration discarded (read by ``extract purge``); an
    interrupted one may still discard more, so its list is refused."""
    data = read_json(manifest)
    if COMPLETED_KEY not in data:
        raise MissingInputError(
            f"{manifest} has no {COMPLETED_KEY}: run `dataset migrate` to the end first"
        )
    return [str(c) for c in data[RERENDER_KEY]]


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


def check_plan_hash(plan: DatasetPlan, ds: DatasetSettings, name: str) -> None:
    """The stored plan must be this content: already migrated, or (for the velocity migration
    only) from before the key existed."""
    known = {ds.content_hash()}
    if name == velocity_migration(ds):
        known.add(ds.content_hash(without=(CONVENTION_KEY,)))
    if plan.config_hash in known:
        return
    velocity = velocity_migration(ds)
    hint = "" if name == velocity else f"; run `physverify dataset migrate --name {velocity}` first"
    raise ConfigError(
        f"dataset {plan.version} was planned with other settings than the current ones "
        f"(beyond velocity_convention): migration refused{hint}"
    )


def baseline(
    plan: DatasetPlan,
    store: DatasetStore,
    ds: DatasetSettings,
    commit: str,
    name: str,
    now: Clock,
) -> dict[str, Any]:
    clip_ids = [v.clip_id for v in plan.variants]
    return {
        "name": name,
        "dataset": DATASET_NAME,
        "version": plan.version,
        "convention": ds.velocity_convention,
        "commit": commit,
        "created_utc": _stamp(now),
        "config_hash": {"old": plan.config_hash, "new": ds.content_hash()},
        "gt_revision": {"old": plan.gt_revision, "new": GT_REVISION},
        COVERS_KEY: covered(plan.gt_revision),
        "rendered_before": [c for c in clip_ids if store.is_done(Step.RENDER, c)],
        "status_before": statuses(plan, store),
        RERENDER_KEY: [],
    }


def load_or_start(
    path: Path,
    plan: DatasetPlan,
    store: DatasetStore,
    ds: DatasetSettings,
    commit: str,
    name: str,
    now: Clock,
) -> dict[str, Any]:
    if not path.is_file():
        started = baseline(plan, store, ds, commit, name, now)
        atomic_write_json(path, started)
        return started
    stored: dict[str, Any] = read_json(path)
    if stored["config_hash"]["new"] != ds.content_hash():
        raise ConfigError(
            f"{path} migrated to config hash {stored['config_hash']['new']}, the current "
            f"settings hash to {ds.content_hash()}: remove the manifest only if that is intended"
        )
    return stored


def rewrite_plan(store: DatasetStore, plan: DatasetPlan, **fields: Any) -> DatasetPlan:
    """The stored plan with ``fields`` replaced."""
    data = plan.to_json() | fields
    atomic_write_json(store.plan_path, data)
    return DatasetPlan.from_json(data)


def reached_apply(plan: DatasetPlan, status_before: Mapping[str, Mapping[str, Any]]) -> DatasetPlan:
    """The part of the plan APPLY had reached before the migration (applied or skipped)."""
    variants = tuple(v for v in plan.variants if status_before[v.clip_id]["state"] != "pending")
    return dataclasses.replace(plan, variants=variants)


def reopen_apply(plan: DatasetPlan, store: DatasetStore) -> int:
    """APPLY pending again for every simulated clip of ``plan`` (skipped ones get a new chance)."""
    reopened = 0
    for variant in plan.variants:
        if not store.is_done(Step.SIMULATE, variant.sim_id):
            continue
        clip_dir = store.clip_dir(variant.clip_id)
        for name in (RECORD, SKIPPED_FILE):
            (clip_dir / name).unlink(missing_ok=True)
        reopened += 1
    return reopened


def reapply(
    store: DatasetStore,
    plugins: Mapping[str, Any],
    ds: DatasetSettings,
    commit: str,
    name: str,
    now: Clock = lambda: dt.datetime.now(dt.UTC),
) -> MigrationReport:
    """Re-run APPLY with this checkout on every clip of the stored plan it reached, logged in the
    manifest of migration ``name``."""
    check_name(name, ds)
    plan = load_plan(store)
    if plan is None:
        raise MissingInputError(f"nothing planned in {store.root}")
    check_plan_hash(plan, ds, name)
    path = manifest_path(store, name)
    if not (path.is_file() and COMPLETED_KEY in read_json(path)):
        check_pending(plan, name)
    manifest = load_or_start(path, plan, store, ds, commit, name, now)
    if COMPLETED_KEY in manifest:
        return _report(path, 0, manifest)
    plan = rewrite_plan(store, plan, config_hash=ds.content_hash())
    reached = reached_apply(plan, manifest["status_before"])
    reopen_apply(reached, store)
    report = apply_variants(reached, store, plugins, ds)
    if store.manifest_path.is_file():
        write_index(store)
    rerender = sorted(
        set(manifest[RERENDER_KEY])
        | {c for c in manifest["rendered_before"] if not store.is_done(Step.RENDER, c)}
    )
    after = statuses(plan, store)
    # The revision is raised only once APPLY has run on every reached clip: a crash before this
    # leaves the old revision, which every build step refuses until the migration is rerun.
    rewrite_plan(store, plan, gt_revision=GT_REVISION)
    manifest |= {
        RERENDER_KEY: rerender,
        "status_after": after,
        "status_changes": status_changes(manifest["status_before"], after),
        COMPLETED_KEY: _stamp(now),
    }
    atomic_write_json(path, manifest)
    return _report(path, len(report.applied), manifest)


def migrate_velocity(
    store: DatasetStore,
    plugins: Mapping[str, Any],
    ds: DatasetSettings,
    commit: str,
    now: Clock = lambda: dt.datetime.now(dt.UTC),
) -> MigrationReport:
    """``reapply`` under ``ds.velocity_convention`` (WP-G 1.C)."""
    return reapply(store, plugins, ds, commit, velocity_migration(ds), now)


def _report(path: Path, applied: int, manifest: Mapping[str, Any]) -> MigrationReport:
    return MigrationReport(path, applied, tuple(manifest[RERENDER_KEY]), manifest["status_changes"])


def _stamp(now: Clock) -> str:
    return now().astimezone(dt.UTC).isoformat(timespec="seconds")
