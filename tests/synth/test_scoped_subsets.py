"""``kubric_synthetic`` declares its check-scoped subsets (``checks_only``) with their exported
clips; the configured ``calib_f3_extra`` adds whole calib scenes to ``full``, scoped to R3b, and
``calib_f3_floor`` is declared alongside it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.datasets._base import ScopedSubset, ScopedSubsetSource
from physverify.datasets._kubric_synthetic import KubricSynthetic
from physverify.schema.clip import ClipManifest, Split, TimeSource
from physverify.schema.io import dumps, envelope
from physverify.synth.generate import plan_subset, save_plan
from physverify.synth.gt.export import DATASET_NAME
from physverify.synth.plan import DatasetPlan, VariantPlan
from physverify.synth.settings import DatasetSettings, dataset_root
from physverify.synth.store import DatasetStore

EXTENSION = "calib_f3_extra"
DECLARED = (EXTENSION, "calib_f3_floor")


def by_name(dataset: KubricSynthetic) -> dict[str, ScopedSubset]:
    return {s.name: s for s in dataset.scoped_subsets()}


def manifest(variant: VariantPlan) -> ClipManifest:
    return ClipManifest(
        dataset=DATASET_NAME,
        clip_id=variant.clip_id,
        split=variant.split,
        video_path=f"{DATASET_NAME}/{variant.clip_id}",
        n_frames=24,
        width=64,
        height=64,
        container_fps=24.0,
        seconds_per_frame=1.0 / 24.0,
        time_source=TimeSource.GROUND_TRUTH,
        scene_id=variant.scene_id,
    )


def exported(settings: Settings, ds: DatasetSettings, plans: list[DatasetPlan]) -> DatasetStore:
    """A store with ``plans`` saved and every planned clip in its index (no clip files)."""
    store = DatasetStore(dataset_root(settings, ds))
    variants = {v.clip_id: v for plan in plans for v in save_plan(store, plan).variants}
    lines = [dumps(envelope(m, m.to_dict())) + "\n" for m in map(manifest, variants.values())]
    store.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    store.manifest_path.write_text("".join(lines), encoding="utf-8")
    return store


def test_the_extension_is_declared_with_its_exported_clips(
    data_settings: Settings, ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    full = plan_subset(ds, families, perturbations, "full")
    extra = plan_subset(ds, families, perturbations, EXTENSION)
    exported(data_settings, ds, [full, extra])
    dataset = KubricSynthetic(data_settings)
    assert isinstance(dataset, ScopedSubsetSource)
    declared = by_name(dataset)
    assert tuple(declared) == DECLARED
    scoped = declared[EXTENSION]
    assert scoped.checks == ("R3b",)
    assert declared["calib_f3_floor"].clips == ()
    assert {c.clip_id for c in scoped.clips} == {v.clip_id for v in extra.variants}
    base_scenes = {v.scene_id for v in full.variants}
    added = [c for c in scoped.clips if c.clip_id not in {v.clip_id for v in full.variants}]
    assert len(added) == 120
    assert not {c.scene_id for c in added} & base_scenes


def test_an_unplanned_extension_has_no_clips(
    data_settings: Settings, ds: DatasetSettings, families: Any, perturbations: Any
) -> None:
    assert KubricSynthetic(data_settings).scoped_subsets() == tuple(
        ScopedSubset(name, ("R3b",), ()) for name in DECLARED
    )
    exported(data_settings, ds, [plan_subset(ds, families, perturbations, "full")])
    assert all(s.clips == () for s in KubricSynthetic(data_settings).scoped_subsets())


def test_a_split_leaves_out_the_clips_only_an_extension_plans(
    data_settings: Settings,
    ds: DatasetSettings,
    families: Any,
    perturbations: Any,
    config_dir: Path,
    environ: dict[str, str],
) -> None:
    full = plan_subset(ds, families, perturbations, "full")
    extra = plan_subset(ds, families, perturbations, EXTENSION)
    exported(data_settings, ds, [full, extra])
    base = {v.clip_id for v in full.variants if v.split is Split.CALIB}
    scenes = {v.scene_id for v in full.variants if v.split is Split.CALIB}
    added = {v.clip_id for v in extra.variants} - {v.clip_id for v in full.variants}

    # Planned counts; the v1-precise store exports 426 / 100 and, with the extension, 540 / 160.
    calib = KubricSynthetic(data_settings).clips("calib")
    assert {c.clip_id for c in calib} == base
    assert {c.scene_id for c in calib} == scenes
    assert (len(base), len(scenes)) == (460, 100)

    env = environ | {"PHYSVERIFY__PATHS__DATA_DIR": str(data_settings.paths.data_dir)}
    off = load_settings(
        config_dir, environ=env, overrides=[f"dataset.subsets.{EXTENSION}.checks_only=[]"]
    )
    unscoped = KubricSynthetic(off).clips("calib")
    assert {c.clip_id for c in unscoped} == base | added
    assert (len(unscoped), len({c.scene_id for c in unscoped})) == (580, 160)
    assert {c.clip_id for c in KubricSynthetic(data_settings).clips(EXTENSION)} == {
        v.clip_id for v in extra.variants
    }
