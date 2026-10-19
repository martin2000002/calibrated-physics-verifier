"""WP-G 1.C: migrating a stored dataset to the position velocity convention equals a fresh build,
discards only the renders whose inputs changed and logs both config hashes."""

from __future__ import annotations

import dataclasses
import datetime as dt
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.schema.states import ObjectStateTable
from physverify.synth.generate import load_plan, plan_subset, save_plan
from physverify.synth.migrate import (
    CONVENTION_KEY,
    RERENDER_KEY,
    migrate_velocity,
    rerender_clip_ids,
)
from physverify.synth.migration_audit import AuditInputs, audit
from physverify.synth.perturb import apply_variants, instantiate
from physverify.synth.plan import DatasetPlan
from physverify.synth.settings import DatasetSettings, load_dataset_settings
from physverify.synth.store import (
    CLOSEST,
    KUBRIC_META,
    LAYERS,
    RECORD,
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
    root: Path, ds: DatasetSettings, plan_hash: str, families: Any, perturbations: Any
) -> tuple[DatasetPlan, DatasetStore, dict[str, Any]]:
    """The smoke subset on the recorded f2 simulation, applied under ``ds``, plan hashed as told."""
    store = DatasetStore(root)
    plan = dataclasses.replace(
        plan_subset(ds, families, perturbations, "smoke"), config_hash=plan_hash
    )
    save_plan(store, plan)
    store.scene_dir(SIM_ID).mkdir(parents=True)
    for path in (DATA / "f2_bounce").iterdir():
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
    for variant in plan.variants:
        for name in FAKE_DOWNSTREAM:
            atomic_write_json(store.clip_dir(variant.clip_id) / name, {"fake": True})
    return plan, store, plugins


def clip_of(plan: DatasetPlan, perturbation: str) -> str:
    return next(v.clip_id for v in plan.variants if v.perturbation == perturbation)


@pytest.fixture
def position_ds(data_settings: Settings) -> DatasetSettings:
    ds = load_dataset_settings(data_settings)
    assert ds.velocity_convention == "position"
    return ds


def test_the_migration_equals_a_fresh_build(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan, old, plugins = legacy(tmp_path, position_ds, families, perturbations)
    migrate_velocity(old, plugins, position_ds, COMMIT, clock)
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
    report = migrate_velocity(store, plugins, position_ds, COMMIT, clock)
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
    plan, store, plugins = legacy(tmp_path, position_ds, families, perturbations)
    first = migrate_velocity(store, plugins, position_ds, COMMIT, clock)
    stamps = {p: p.stat().st_mtime_ns for p in store.root.rglob("*") if p.is_file()}
    again = migrate_velocity(store, plugins, position_ds, COMMIT, clock)
    assert (again.applied, again.rerender) == (0, first.rerender)
    assert {p: p.stat().st_mtime_ns for p in store.root.rglob("*") if p.is_file()} == stamps


def test_a_crashed_migration_still_lists_the_render_it_discarded(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan, store, plugins = legacy(tmp_path, position_ds, families, perturbations)
    report = migrate_velocity(store, plugins, position_ds, COMMIT, clock)
    manifest = read_json(report.manifest)
    del manifest["completed_utc"]
    atomic_write_json(report.manifest, manifest | {RERENDER_KEY: []})
    again = migrate_velocity(store, plugins, position_ds, COMMIT, clock)
    assert again.rerender == report.rerender


def test_a_plan_with_other_content_is_refused(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    other = position_ds.model_copy(update={"seed": position_ds.seed + 1})
    _, store, plugins = build(
        tmp_path / "other", other, other.content_hash(), families, perturbations
    )
    with pytest.raises(ConfigError, match="migration refused"):
        migrate_velocity(store, plugins, position_ds, COMMIT, clock)
    assert not (store.root / "migrations").exists()


def test_the_audit_compares_the_migration_with_the_copy_taken_before(
    tmp_path: Path, position_ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan, store, plugins = legacy(tmp_path, position_ds, families, perturbations)
    before = DatasetStore(tmp_path / "before")
    shutil.copytree(store.root, before.root)
    report = migrate_velocity(store, plugins, position_ds, COMMIT, clock)
    inputs = AuditInputs(plan, store, before, report.manifest, plugins, position_ds)
    found = audit(inputs, [tmp_path / "cache"], [])
    edited = clip_of(plan, "r3_restitution_gain")
    assert found["rerender"] == {
        "observed": [edited],
        "predicted_always": [edited],
        "missing_from_always": [],
        "conditional_rerendered": [],
        "unexpected": [],
    }
    assert found["media"] == {"kept": 1, "frames_differ": [], "mp4_differ": []}
    assert found["contacts"]["fixed_changed"] == []
    assert found["labels"]["r3_off_nominal"] == {}
    assert found["rest"]["other_cells"] == []
    assert found["cache"]["stale_meta"] == {}
