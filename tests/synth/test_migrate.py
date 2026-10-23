"""Migrating a stored dataset to the position velocity convention equals a fresh build,
discards only the renders whose inputs changed and logs both config hashes. The
floor-crossing migration re-solves a penetrated bounce row, raises the plan's ``gt_revision`` and
re-renders only the clip whose flight moved."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from physverify.config.models import Settings
from physverify.core.errors import ConfigError, MissingInputError, UsageError
from physverify.schema.scene import PerturbationRecord
from physverify.schema.states import ObjectStateTable
from physverify.synth import magnitudes, migrate
from physverify.synth.ballistics import time_to_height
from physverify.synth.generate import load_plan, plan_subset, save_plan
from physverify.synth.gt_export import read_index
from physverify.synth.migrate import (
    COMPLETED_KEY,
    CONVENTION_KEY,
    COVERS_KEY,
    FLOOR_CROSSING,
    PAIR_COLLISION,
    RERENDER_KEY,
    MigrationReport,
    manifest_path,
    reapply,
    rerender_clip_ids,
    velocity_migration,
)
from physverify.synth.perturb import apply_variants, instantiate
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.plan import GT_MIGRATIONS, GT_REVISION, DatasetPlan
from physverify.synth.settings import DatasetSettings, load_dataset_settings
from physverify.synth.store import (
    CLOSEST,
    FRAMES_DIR,
    GT_META,
    KUBRIC_META,
    LAYERS,
    RECORD,
    SIM_STATES,
    SKIPPED_FILE,
    VARIANT_STATES,
    DatasetStore,
    Step,
    atomic_write_json,
    read_json,
)

DATA = Path(__file__).parent / "data"
SIM_ID = "f2_bounce-s000"
COMMIT = "c0ffee"
FAKE_DOWNSTREAM = (KUBRIC_META, LAYERS, CLOSEST)


def clock() -> dt.datetime:
    return dt.datetime(2026, 10, 18, tzinfo=dt.UTC)


def build(
    root: Path,
    ds: DatasetSettings,
    plan_hash: str,
    families: Any,
    perturbations: Any,
    simulation: Path = DATA / "clean" / "f2_bounce",
) -> tuple[DatasetPlan, DatasetStore, dict[str, Any]]:
    """The smoke subset on the recorded f2 simulation, applied under ``ds``, plan hashed as told."""
    store = DatasetStore(root)
    plan = dataclasses.replace(
        plan_subset(ds, families, perturbations, "smoke"), config_hash=plan_hash
    )
    save_plan(store, plan)
    store.scene_dir(SIM_ID).mkdir(parents=True)
    for path in simulation.iterdir():
        shutil.copy(path, store.scene_dir(SIM_ID) / path.name)
    plugins = instantiate(perturbations, ds)
    apply_variants(plan, store, plugins, ds)
    return plan, store, plugins


def legacy(
    tmp_path: Path, ds: DatasetSettings, families: Any, perturbations: Any
) -> tuple[DatasetPlan, DatasetStore, dict[str, Any]]:
    """A store built before ``velocity_convention`` existed, every clip rendered."""
    state = ds.model_copy(update={CONVENTION_KEY: "state"})
    old_hash = ds.content_hash(without=(CONVENTION_KEY,))
    plan, store, plugins = build(tmp_path / "old", state, old_hash, families, perturbations)
    rendered(plan, store)
    return plan, store, plugins


def rendered(plan: DatasetPlan, store: DatasetStore) -> None:
    for variant in plan.variants:
        for name in FAKE_DOWNSTREAM:
            atomic_write_json(store.clip_dir(variant.clip_id) / name, {"fake": True})


def clip_of(plan: DatasetPlan, perturbation: str) -> str:
    return next(v.clip_id for v in plan.variants if v.perturbation == perturbation)


def reapply_velocity(
    store: DatasetStore,
    plugins: Mapping[str, Any],
    ds: DatasetSettings,
    commit: str,
    now: Callable[[], dt.datetime],
) -> MigrationReport:
    return reapply(store, plugins, ds, commit, velocity_migration(ds), now)


@pytest.fixture
def position_ds(data_settings: Settings) -> DatasetSettings:
    ds = load_dataset_settings(data_settings)
    assert ds.velocity_convention == "position"
    return ds


def test_the_migration_equals_a_fresh_build(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan, old, plugins = legacy(tmp_path, position_ds, families, perturbations)
    reapply_velocity(old, plugins, position_ds, COMMIT, clock)
    _, fresh, _ = build(
        tmp_path / "fresh", position_ds, position_ds.content_hash(), families, perturbations
    )
    for variant in plan.variants:
        migrated = ObjectStateTable.load_npz(old.clip_dir(variant.clip_id) / VARIANT_STATES)
        built = ObjectStateTable.load_npz(fresh.clip_dir(variant.clip_id) / VARIANT_STATES)
        for field in dataclasses.fields(built):
            a, b = getattr(migrated, field.name), getattr(built, field.name)
            same = (
                np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
                if isinstance(a, np.ndarray)
                else a == b
            )
            assert same, (variant.clip_id, field.name)
        assert read_json(old.clip_dir(variant.clip_id) / RECORD) == read_json(
            fresh.clip_dir(variant.clip_id) / RECORD
        )
    assert load_plan(old) == dataclasses.replace(
        load_plan(fresh) or plan, config_hash=position_ds.content_hash()
    )


def test_the_manifest_lists_the_discarded_renders_and_both_hashes(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan, store, plugins = legacy(tmp_path, position_ds, families, perturbations)
    old_hash = plan.config_hash
    report = reapply_velocity(store, plugins, position_ds, COMMIT, clock)
    unedited, edited = clip_of(plan, "none"), clip_of(plan, "r3_restitution_gain")
    assert report.rerender == (edited,)
    assert rerender_clip_ids(report.manifest) == [edited]
    manifest = read_json(report.manifest)
    assert manifest["commit"] == COMMIT and manifest["convention"] == "position"
    assert manifest["config_hash"] == {"old": old_hash, "new": position_ds.content_hash()}
    assert manifest["status_changes"] == {}
    assert sorted(manifest["rendered_before"]) == sorted(v.clip_id for v in plan.variants)
    assert read_json(store.plan_path)["config_hash"] == position_ds.content_hash()
    for name in FAKE_DOWNSTREAM:
        assert (store.clip_dir(unedited) / name).is_file(), name
        assert not (store.clip_dir(edited) / name).exists(), name
    assert all(store.is_done(Step.APPLY, v.clip_id) for v in plan.variants)


def test_a_completed_migration_runs_again_as_a_no_op(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    _plan, store, plugins = legacy(tmp_path, position_ds, families, perturbations)
    first = reapply_velocity(store, plugins, position_ds, COMMIT, clock)
    stamps = {p: p.stat().st_mtime_ns for p in store.root.rglob("*") if p.is_file()}
    again = reapply_velocity(store, plugins, position_ds, COMMIT, clock)
    assert (again.applied, again.rerender) == (0, first.rerender)
    assert {p: p.stat().st_mtime_ns for p in store.root.rglob("*") if p.is_file()} == stamps


def test_a_crashed_migration_still_lists_the_render_it_discarded(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    _plan, store, plugins = legacy(tmp_path, position_ds, families, perturbations)
    report = reapply_velocity(store, plugins, position_ds, COMMIT, clock)
    manifest = read_json(report.manifest)
    del manifest[COMPLETED_KEY]
    atomic_write_json(report.manifest, manifest | {RERENDER_KEY: []})
    with pytest.raises(MissingInputError, match=COMPLETED_KEY):
        rerender_clip_ids(report.manifest)
    again = reapply_velocity(store, plugins, position_ds, COMMIT, clock)
    assert again.rerender == report.rerender


def test_a_clip_apply_never_reached_stays_pending(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan, store, plugins = legacy(tmp_path, position_ds, families, perturbations)
    pending = clip_of(plan, "r3_restitution_gain")
    shutil.rmtree(store.clip_dir(pending))
    report = reapply_velocity(store, plugins, position_ds, COMMIT, clock)
    assert report.applied == len(plan.variants) - 1
    assert not store.is_done(Step.APPLY, pending)
    assert report.status_changes == {} and report.rerender == ()


def test_a_plan_with_other_content_is_refused(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    other = position_ds.model_copy(update={"seed": position_ds.seed + 1})
    _, store, plugins = build(
        tmp_path / "other", other, other.content_hash(), families, perturbations
    )
    with pytest.raises(ConfigError, match="migration refused"):
        reapply_velocity(store, plugins, position_ds, COMMIT, clock)
    assert not (store.root / "migrations").exists()


class Refusing:
    """A perturbation the new settings no longer apply to this scene."""

    name = "none"
    target = PerturbationTarget.TRAJECTORY

    def applicable(self, scene: Any) -> bool:
        return False


def test_a_clip_now_refused_loses_its_render_and_export(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan, store, plugins = legacy(tmp_path, position_ds, families, perturbations)
    refused = clip_of(plan, "none")
    clip_dir = store.clip_dir(refused)
    atomic_write_json(clip_dir / GT_META, {"fake": True})
    (clip_dir / FRAMES_DIR).mkdir()
    (clip_dir / FRAMES_DIR / "000000.png").write_bytes(b"old")
    store.manifest_path.write_text("stale\n", encoding="utf-8")
    report = reapply_velocity(store, plugins | {"none": Refusing()}, position_ds, COMMIT, clock)
    assert sorted(p.name for p in clip_dir.iterdir()) == [SKIPPED_FILE]
    assert refused in report.rerender
    assert report.status_changes[refused]["before"]["state"] == "applied"
    assert report.status_changes[refused]["after"]["state"] == "skipped"
    assert refused in rerender_clip_ids(report.manifest)
    assert read_index(store) == []


def legacy_crossing(
    states: ObjectStateTable, index: int, k: int, radius: float, gz: float
) -> tuple[int, float] | None:
    """``floor_crossing`` before the fix: a row already under the floor fell back to t_c = t_k."""
    t_k = float(states.t[k])
    z, vz = float(states.position[k, index, 2]), float(states.velocity[k, index, 2])
    tau = time_to_height(z, vz, gz, radius)
    t_c = t_k + tau if math.isfinite(tau) else t_k
    return None if t_c > float(states.t[k + 1]) else (k, t_c)


def penetrated_simulation(root: Path) -> Path:
    """The recorded f2 simulation with the first row after the ball's bounce replaced by its
    incoming arc continued from the row before: under the floor and still falling."""
    root.mkdir(parents=True)
    for path in (DATA / "clean" / "f2_bounce").iterdir():
        shutil.copy(path, root / path.name)
    states = ObjectStateTable.load_npz(root / SIM_STATES)
    index = states.object_index("ball_0")
    k = 6
    assert states.velocity[k - 1, index, 2] < 0.0 < states.velocity[k, index, 2]
    dt_s = float(states.t[k] - states.t[k - 1])
    g = np.array([0.0, 0.0, -9.81])
    position, velocity = states.position.copy(), states.velocity.copy()
    p, v = position[k - 1, index], velocity[k - 1, index]
    position[k, index] = p + v * dt_s + 0.5 * g * dt_s**2
    velocity[k, index] = v + g * dt_s
    dataclasses.replace(states, position=position, velocity=velocity).save_npz(root / SIM_STATES)
    return root


def before_floor_crossing(
    tmp_path: Path,
    ds: DatasetSettings,
    families: Any,
    perturbations: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[DatasetPlan, DatasetStore, dict[str, Any]]:
    """A store applied and rendered with the old bounce solver, its plan from before
    ``gt_revision`` existed."""
    simulation = penetrated_simulation(tmp_path / "sim")
    with monkeypatch.context() as patch:
        patch.setattr(magnitudes, "floor_crossing", legacy_crossing)
        plan, store, plugins = build(
            tmp_path / "old", ds, ds.content_hash(), families, perturbations, simulation
        )
    rendered(plan, store)
    stored = read_json(store.plan_path)
    del stored["gt_revision"]
    atomic_write_json(store.plan_path, stored)
    return plan, store, plugins


def test_the_floor_crossing_migration_rerenders_only_the_moved_flight(
    tmp_path: Path,
    position_ds: DatasetSettings,
    families: Any,
    perturbations: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, store, plugins = before_floor_crossing(
        tmp_path, position_ds, families, perturbations, monkeypatch
    )
    old_frame = PerturbationRecord.load_json(
        store.clip_dir(clip_of(plan, "r3_restitution_gain")) / RECORD
    )
    report = reapply(store, plugins, position_ds, COMMIT, FLOOR_CROSSING, clock)
    unedited, edited = clip_of(plan, "none"), clip_of(plan, "r3_restitution_gain")
    assert report.rerender == (edited,)
    assert report.manifest == manifest_path(store, FLOOR_CROSSING)
    manifest = read_json(report.manifest)
    assert manifest["name"] == FLOOR_CROSSING
    assert manifest["gt_revision"] == {"old": 0, "new": GT_REVISION}
    assert not manifest_path(store, "velocity-position").exists()
    stored = load_plan(store)
    assert stored is not None and stored.gt_revision == GT_REVISION
    for name in FAKE_DOWNSTREAM:
        assert (store.clip_dir(unedited) / name).is_file(), name
        assert not (store.clip_dir(edited) / name).exists(), name
    new_frame = PerturbationRecord.load_json(store.clip_dir(edited) / RECORD)
    assert float(old_frame.params["impact_frame"]) == 6.0
    assert 5.0 < float(new_frame.params["impact_frame"]) < 6.0

    stamps = {p: p.stat().st_mtime_ns for p in store.root.rglob("*") if p.is_file()}
    again = reapply(store, plugins, position_ds, COMMIT, FLOOR_CROSSING, clock)
    assert (again.applied, again.rerender) == (0, report.rerender)
    assert {p: p.stat().st_mtime_ns for p in store.root.rglob("*") if p.is_file()} == stamps


def test_a_migration_cut_short_keeps_the_old_gt_revision(
    tmp_path: Path,
    position_ds: DatasetSettings,
    families: Any,
    perturbations: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, store, plugins = before_floor_crossing(
        tmp_path, position_ds, families, perturbations, monkeypatch
    )

    def crash(*args: Any) -> None:
        apply_variants(*args)
        raise RuntimeError("killed")

    with monkeypatch.context() as patch:
        patch.setattr(migrate, "apply_variants", crash)
        with pytest.raises(RuntimeError, match="killed"):
            reapply(store, plugins, position_ds, COMMIT, FLOOR_CROSSING, clock)
    assert COMPLETED_KEY not in read_json(manifest_path(store, FLOOR_CROSSING))
    cut = load_plan(store)
    assert cut is not None and cut.gt_revision == 0
    with pytest.raises(ConfigError, match=f"migrate --name {FLOOR_CROSSING}"):
        cut.check_gt_revision()
    report = reapply(store, plugins, position_ds, COMMIT, FLOOR_CROSSING, clock)
    assert report.rerender == (clip_of(plan, "r3_restitution_gain"),)
    done = load_plan(store)
    assert done is not None and done.gt_revision == GT_REVISION


def test_a_migration_cut_after_raising_the_revision_finishes_on_rerun(
    tmp_path: Path,
    position_ds: DatasetSettings,
    families: Any,
    perturbations: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, store, plugins = before_floor_crossing(
        tmp_path, position_ds, families, perturbations, monkeypatch
    )

    def crash(*args: Any) -> None:
        raise RuntimeError("killed")

    with monkeypatch.context() as patch:
        patch.setattr(migrate, "status_changes", crash)
        with pytest.raises(RuntimeError, match="killed"):
            reapply(store, plugins, position_ds, COMMIT, FLOOR_CROSSING, clock)
    cut = load_plan(store)
    assert cut is not None and cut.gt_revision == GT_REVISION
    assert COMPLETED_KEY not in read_json(manifest_path(store, FLOOR_CROSSING))
    report = reapply(store, plugins, position_ds, COMMIT, FLOOR_CROSSING, clock)
    assert report.rerender == (clip_of(plan, "r3_restitution_gain"),)
    assert COMPLETED_KEY in read_json(manifest_path(store, FLOOR_CROSSING))


def test_the_floor_crossing_migration_needs_the_velocity_one_first(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    _, store, plugins = legacy(tmp_path, position_ds, families, perturbations)
    with pytest.raises(ConfigError, match="migrate --name velocity-position"):
        reapply(store, plugins, position_ds, COMMIT, FLOOR_CROSSING, clock)
    assert not (store.root / "migrations").exists()


def test_a_plan_from_an_older_gt_revision_is_refused_until_migrated(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan, store, plugins = build(
        tmp_path / "store", position_ds, position_ds.content_hash(), families, perturbations
    )
    assert plan.gt_revision == GT_REVISION
    assert read_json(store.plan_path)["gt_revision"] == GT_REVISION
    stored = read_json(store.plan_path)
    del stored["gt_revision"]
    atomic_write_json(store.plan_path, stored)
    with pytest.raises(ConfigError, match=f"migrate --name {FLOOR_CROSSING}"):
        save_plan(store, plan)
    reapply(store, plugins, position_ds, COMMIT, FLOOR_CROSSING, clock)
    assert save_plan(store, plan).gt_revision == GT_REVISION


def test_a_plan_from_a_newer_gt_revision_asks_for_a_newer_checkout(
    position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan = plan_subset(position_ds, families, perturbations, "smoke")
    newer = dataclasses.replace(plan, gt_revision=GT_REVISION + 1)
    with pytest.raises(ConfigError, match="update the checkout"):
        plan.merged(newer)
    assert DatasetPlan.from_json(newer.to_json()) == newer


def test_an_unknown_migration_name_is_a_usage_error(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    _, store, plugins = build(
        tmp_path / "store", position_ds, position_ds.content_hash(), families, perturbations
    )
    with pytest.raises(UsageError, match="unknown migration"):
        reapply(store, plugins, position_ds, COMMIT, "velocity-state", clock)


def test_the_revision_is_the_length_of_the_migration_table(
    position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    assert len(GT_MIGRATIONS) == GT_REVISION and GT_MIGRATIONS[0] == FLOOR_CROSSING
    assert migrate.migration_names(position_ds) == ("velocity-position", *GT_MIGRATIONS)
    plan = plan_subset(position_ds, families, perturbations, "smoke")
    assert plan.gt_revision == GT_REVISION and plan.pending_migration() is None
    for revision, name in enumerate(GT_MIGRATIONS):
        old = dataclasses.replace(plan, gt_revision=revision)
        assert old.pending_migration() == name
        with pytest.raises(ConfigError, match=f"migrate --name {name}`"):
            old.check_gt_revision()


def test_a_migration_that_is_not_the_pending_one_is_refused(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    _, store, plugins = build(
        tmp_path / "store", position_ds, position_ds.content_hash(), families, perturbations
    )
    for name in GT_MIGRATIONS:
        with pytest.raises(ConfigError, match="is up to date"):
            reapply(store, plugins, position_ds, COMMIT, name, clock)
    assert not (store.root / "migrations").exists()


def test_the_first_fix_from_revision_zero_covers_every_fix(
    tmp_path: Path,
    position_ds: DatasetSettings,
    families: Any,
    perturbations: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, store, plugins = before_floor_crossing(
        tmp_path, position_ds, families, perturbations, monkeypatch
    )
    report = reapply(store, plugins, position_ds, COMMIT, GT_MIGRATIONS[0], clock)
    manifest = read_json(report.manifest)
    assert manifest["gt_revision"] == {"old": 0, "new": GT_REVISION}
    assert manifest[COVERS_KEY] == list(GT_MIGRATIONS)
    stored = load_plan(store)
    assert stored is not None and stored.gt_revision == GT_REVISION
    for name in GT_MIGRATIONS[1:]:
        with pytest.raises(ConfigError, match="is up to date"):
            reapply(store, plugins, position_ds, COMMIT, name, clock)


def test_the_velocity_migration_records_what_it_covers(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    _plan, store, plugins = legacy(tmp_path, position_ds, families, perturbations)
    report = reapply_velocity(store, plugins, position_ds, COMMIT, clock)
    assert read_json(report.manifest)[COVERS_KEY] == []


def test_a_later_fix_on_a_plan_that_lacks_an_earlier_one_is_refused(
    tmp_path: Path,
    position_ds: DatasetSettings,
    families: Any,
    perturbations: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, store, plugins = before_floor_crossing(
        tmp_path, position_ds, families, perturbations, monkeypatch
    )
    with pytest.raises(ConfigError, match=f"needs `{FLOOR_CROSSING}` first"):
        reapply(store, plugins, position_ds, COMMIT, PAIR_COLLISION, clock)
    assert not (store.root / "migrations").exists()
