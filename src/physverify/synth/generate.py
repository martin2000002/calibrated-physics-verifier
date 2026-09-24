"""The ``generate`` step: plan a subset (scenes, splits, variants) and write every base scene spec
plus the re-simulated specs of scene perturbations."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from physverify.config.models import SizePrior
from physverify.core.errors import ConfigError
from physverify.schema.clip import Split
from physverify.schema.scene import SceneSpec
from physverify.synth.perturb import REFUSALS, SeedKey, perturb_scene
from physverify.synth.perturbations._base import PerturbationTarget
from physverify.synth.plan import NONE, DatasetPlan, ScenePlan, VariantPlan, clip_id_of, scene_id_of
from physverify.synth.seeds import is_calib, scene_rng, scene_seed, split_phase
from physverify.synth.settings import (
    DatasetSettings,
    FamilyClass,
    PerturbationClass,
    SubsetSettings,
    family_params,
    perturbation_params,
)
from physverify.synth.store import (
    SCENE_RECORD,
    SPEC,
    DatasetStore,
    Step,
    atomic_save,
    atomic_write_json,
    read_json,
)


def subset_settings(ds: DatasetSettings, subset: str) -> SubsetSettings:
    if subset not in ds.subsets:
        raise ConfigError(f"unknown subset {subset!r}; configured: {sorted(ds.subsets)}")
    return ds.subsets[subset]


def check_subset(
    sub: SubsetSettings,
    families: Mapping[str, FamilyClass],
    perturbations: Mapping[str, PerturbationClass],
) -> None:
    """Every family and perturbation named by the subset exists and applies to that family."""
    for family, names in sub.perturbations.items():
        if family not in families:
            raise ConfigError(f"unknown family {family!r}; registered: {sorted(families)}")
        for name in names:
            if name not in perturbations:
                raise ConfigError(
                    f"unknown perturbation {name!r}; registered: {sorted(perturbations)}"
                )
            allowed = getattr(perturbations[name], "families", ())
            if allowed and family not in allowed:
                raise ConfigError(f"{name} does not apply to {family} (only {list(allowed)})")


def split_of(ds: DatasetSettings, family: str, index: int) -> Split:
    calib = is_calib(index, ds.calib_fraction, split_phase(ds.seed, family))
    return Split.CALIB if calib else Split.EVAL


def variant_levels(sub: SubsetSettings, index: int, position: int) -> tuple[int, ...]:
    """Every configured level, or one level cycling with scene index and perturbation position."""
    if sub.level_mode == "all":
        return sub.levels
    return (sub.levels[(index + position) % len(sub.levels)],)


def scene_variants(
    scene: ScenePlan,
    sub: SubsetSettings,
    perturbations: Mapping[str, PerturbationClass],
    ds: DatasetSettings,
) -> list[VariantPlan]:
    out = [_variant(scene, clip_id_of(scene.scene_id, NONE, 0), scene.scene_id, NONE, 0, 0.0)]
    for position, name in enumerate(sorted(sub.perturbations[scene.family])):
        plugin = perturbations[name]
        levels = perturbation_params(plugin, ds).levels
        for level in variant_levels(sub, scene.index, position):
            clip_id = clip_id_of(scene.scene_id, name, level)
            scene_level = plugin.target is PerturbationTarget.SCENE
            sim_id = clip_id if scene_level else scene.scene_id
            out.append(_variant(scene, clip_id, sim_id, name, level, float(levels[level - 1])))
    return out


def _variant(
    scene: ScenePlan, clip_id: str, sim_id: str, perturbation: str, level: int, target: float
) -> VariantPlan:
    return VariantPlan(
        clip_id=clip_id,
        scene_id=scene.scene_id,
        sim_id=sim_id,
        family=scene.family,
        index=scene.index,
        perturbation=perturbation,
        level=level,
        target_magnitude=target,
        split=scene.split,
    )


def plan_subset(
    ds: DatasetSettings,
    families: Mapping[str, FamilyClass],
    perturbations: Mapping[str, PerturbationClass],
    subset: str,
) -> DatasetPlan:
    sub = subset_settings(ds, subset)
    check_subset(sub, families, perturbations)
    scenes = [
        ScenePlan(scene_id_of(family, index), family, index, split_of(ds, family, index))
        for family in sorted(sub.perturbations)
        for index in range(sub.scenes_per_family)
    ]
    variants = [v for scene in scenes for v in scene_variants(scene, sub, perturbations, ds)]
    return DatasetPlan(
        version=ds.version,
        config_hash=ds.content_hash(),
        scenes=tuple(scenes),
        variants=tuple(variants),
        subsets={subset: tuple(sorted(v.clip_id for v in variants))},
    )


def load_plan(store: DatasetStore) -> DatasetPlan | None:
    return DatasetPlan.from_json(read_json(store.plan_path)) if store.plan_path.is_file() else None


def save_plan(store: DatasetStore, plan: DatasetPlan) -> DatasetPlan:
    """Merge ``plan`` into the stored one (same version and content hash) and write it."""
    known = load_plan(store)
    merged = plan if known is None else known.merged(plan)
    atomic_write_json(store.plan_path, merged.to_json())
    return merged


def base_scene(
    scene: ScenePlan, family: FamilyClass, ds: DatasetSettings, priors: Mapping[str, SizePrior]
) -> SceneSpec:
    params = family_params(family, ds, dict(priors))
    draft = family().sample(scene_rng(ds.seed, scene.family, scene.index), params)
    return dataclasses.replace(
        draft, scene_id=scene.scene_id, seed=scene_seed(ds.seed, scene.family, scene.index)
    )


def write_specs(
    plan: DatasetPlan,
    store: DatasetStore,
    families: Mapping[str, FamilyClass],
    plugins: Mapping[str, Any],
    ds: DatasetSettings,
    priors: Mapping[str, SizePrior],
) -> list[str]:
    """Writes the missing specs; returns the sim_ids written. Refused scene edits are skipped."""
    written: list[str] = []
    for scene in plan.scenes:
        if not store.is_done(Step.GENERATE, scene.scene_id):
            atomic_save(
                base_scene(scene, families[scene.family], ds, priors),
                store.scene_dir(scene.scene_id) / SPEC,
            )
            written.append(scene.scene_id)
    for variant in plan.variants:
        if not variant.scene_perturbation or store.is_done(Step.GENERATE, variant.sim_id):
            continue
        if store.is_skipped(variant.clip_id):
            continue
        if _write_scene_variant(variant, store, plugins[variant.perturbation], ds):
            written.append(variant.sim_id)
    return written


def _write_scene_variant(
    variant: VariantPlan, store: DatasetStore, plugin: Any, ds: DatasetSettings
) -> bool:
    scene = SceneSpec.load_json(store.scene_dir(variant.scene_id) / SPEC)
    key = SeedKey(ds.seed, variant.family, variant.index, variant.level)
    try:
        edited, record = perturb_scene(plugin, scene, variant.target_magnitude, key)
    except REFUSALS as exc:
        store.mark_skipped(variant.clip_id, str(exc))
        return False
    sim_dir = store.scene_dir(variant.sim_id)
    atomic_save(record, sim_dir / SCENE_RECORD)
    atomic_save(edited, sim_dir / SPEC)
    return True
