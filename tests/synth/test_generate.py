"""Planning: subset sizes, levels, splits, determinism and resumable spec writing."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from physverify.core.errors import ConfigError
from physverify.schema.scene import SceneSpec
from physverify.synth.generate import (
    load_plan,
    plan_subset,
    save_plan,
    subset_settings,
    write_specs,
)
from physverify.synth.plan import NONE
from physverify.synth.settings import DatasetSettings, SubsetSettings
from physverify.synth.store import SPEC, DatasetStore, Step

# Deliverable 1 runs `local` on the laptop; `full` is generated on the H200.
EXPECTED = {"smoke": (2, 1), "local": (23, 7), "full": (920, 280)}


@pytest.mark.parametrize(("subset", "sizes"), sorted(EXPECTED.items()))
def test_subset_sizes(
    ds: DatasetSettings, families: Any, perturbations: Any, subset: str, sizes: tuple[int, int]
) -> None:
    plan = plan_subset(ds, families, perturbations, subset)
    assert (len(plan.variants), len(plan.sim_ids())) == sizes
    assert len({v.clip_id for v in plan.variants}) == sizes[0]


def test_local_is_one_scene_per_family_and_one_mid_level_of_every_full_perturbation(
    ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    local = plan_subset(ds, families, perturbations, "local")
    full = subset_settings(ds, "full")
    assert Counter(s.family for s in local.scenes) == dict.fromkeys(full.perturbations, 1)
    edits = {(v.family, v.perturbation) for v in local.variants if v.perturbation != NONE}
    assert edits == {(f, p) for f, names in full.perturbations.items() for p in names}
    assert {v.level for v in local.variants if v.perturbation != NONE} == {3}


def test_full_cycles_levels_and_balances_splits(
    ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan = plan_subset(ds, families, perturbations, "full")
    per_edit = Counter((v.family, v.perturbation, v.level) for v in plan.variants)
    assert {n for (_, name, _), n in per_edit.items() if name != NONE} == {10}
    calib = Counter(s.family for s in plan.scenes if s.split.value == "calib")
    assert all(abs(n - 20) <= 1 for n in calib.values())


def test_scene_perturbations_get_their_own_simulation(
    ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan = plan_subset(ds, families, perturbations, "local")
    own = {v.perturbation for v in plan.variants if v.scene_perturbation}
    assert own == {"r5b_gravity_scale"}
    assert all(v.sim_id == v.scene_id for v in plan.variants if not v.scene_perturbation)


def test_target_magnitude_comes_from_the_plugin_levels(
    ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    plan = plan_subset(ds, families, perturbations, "smoke")
    edit = next(v for v in plan.variants if v.perturbation == "r3_restitution_gain")
    assert edit.level == 4
    assert edit.target_magnitude == pytest.approx(0.5)


def test_plans_are_deterministic_and_merge(
    ds: DatasetSettings, families: Any, perturbations: Any, store: DatasetStore
) -> None:
    first = plan_subset(ds, families, perturbations, "local")
    assert first.to_json() == plan_subset(ds, families, perturbations, "local").to_json()
    save_plan(store, plan_subset(ds, families, perturbations, "smoke"))
    merged = save_plan(store, first)
    assert set(merged.subsets) == {"smoke", "local"}
    reloaded = load_plan(store)
    assert reloaded is not None and reloaded.to_json() == merged.to_json()
    assert len(reloaded.subset("smoke").variants) == 2


def test_changed_content_needs_a_new_version(
    ds: DatasetSettings, families: Any, perturbations: Any, store: DatasetStore
) -> None:
    save_plan(store, plan_subset(ds, families, perturbations, "smoke"))
    changed = ds.model_copy(update={"seed": ds.seed + 1})
    with pytest.raises(ConfigError, match="version"):
        save_plan(store, plan_subset(changed, families, perturbations, "smoke"))


def test_unknown_subsets_and_perturbations_are_config_errors(
    ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    with pytest.raises(ConfigError):
        subset_settings(ds, "nope")
    bad = SubsetSettings(scenes_per_family=1, levels=(1,), perturbations={"f2_bounce": ("r9",)})
    changed = ds.model_copy(update={"subsets": {**ds.subsets, "bad": bad}})
    with pytest.raises(ConfigError, match="r9"):
        plan_subset(changed, families, perturbations, "bad")


def test_write_specs_is_resumable_and_seeded(
    ds: DatasetSettings,
    families: Any,
    plugins: dict[str, Any],
    perturbations: Any,
    priors: dict[str, Any],
    store: DatasetStore,
) -> None:
    plan = plan_subset(ds, families, perturbations, "local")
    written = write_specs(plan, store, families, plugins, ds, priors)
    assert sorted(written) == sorted(plan.sim_ids())
    assert all(store.is_done(Step.GENERATE, s) for s in plan.sim_ids())
    first = SceneSpec.load_json(store.scene_dir("f2_bounce-s000") / SPEC)
    assert write_specs(plan, store, families, plugins, ds, priors) == []
    (store.scene_dir("f2_bounce-s000") / SPEC).unlink()
    write_specs(plan, store, families, plugins, ds, priors)
    assert SceneSpec.load_json(store.scene_dir("f2_bounce-s000") / SPEC) == first
