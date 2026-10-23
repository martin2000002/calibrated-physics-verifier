"""The fresh Kubric renders of the final looks: their dataset overrides and their stored plan."""

from __future__ import annotations

from physverify.config.models import Settings
from physverify.datasets.kubric_fresh import KubricFreshSettings
from physverify.freeze.errors import QueryError
from physverify.schema.clip import Split
from physverify.synth.generate import load_plan
from physverify.synth.settings import dataset_root, load_dataset_settings
from physverify.synth.store import DatasetStore

FRESH_SUBSET = "full"
VERSION_ENV = "PHYSVERIFY__DATASET__VERSION"


def render_overrides(fresh: KubricFreshSettings, dataset: str, seed: int) -> dict[str, str]:
    """The dataset section a fresh render reads: its version, seed and scenes, with no calib
    clip."""
    extra: dict[str, str]
    if dataset == "kubric_fresh_ood":
        version, scenes = fresh.ood_version, fresh.ood_scenes_per_family
        extra = {"PHYSVERIFY__DATASET__HDRI_POOL": fresh.ood_hdri_pool}
    elif dataset in ("kubric_fresh", "kubric_fresh_blind"):
        version, scenes, extra = fresh.version, fresh.scenes_per_family, {}
    else:
        raise QueryError(f"{dataset} is not a fresh Kubric set")
    return {
        VERSION_ENV: version,
        "PHYSVERIFY__DATASET__SEED": str(seed),
        "PHYSVERIFY__DATASET__CALIB_FRACTION": "0.0",
        f"PHYSVERIFY__DATASET__SUBSETS__{FRESH_SUBSET.upper()}__SCENES_PER_FAMILY": str(scenes),
        **extra,
    }


def plan_lines(settings: Settings, version: str, seed: int) -> list[str]:
    """The stored plan is the render of ``seed`` (its content hash), with no calib clip."""
    ds = load_dataset_settings(settings)
    if (ds.version, ds.seed) != (version, seed):
        raise QueryError(
            f"the render settings give {ds.version} seed {ds.seed}, not {version} {seed}"
        )
    store = DatasetStore(dataset_root(settings, ds))
    plan = load_plan(store)
    if plan is None:
        raise QueryError(f"no plan at {store.plan_path}: render first")
    if plan.version != version or plan.config_hash != ds.content_hash():
        raise QueryError(f"{store.plan_path} is not the render of {version} seed {seed}")
    calib = [s.scene_id for s in plan.scenes if s.split is Split.CALIB]
    calib += [v.clip_id for v in plan.variants if v.split is Split.CALIB]
    if calib:
        raise QueryError(f"{store.plan_path} plans {len(calib)} calib scenes or clips")
    return [
        (
            f"plan {store.plan_path}: version {version}, seed {seed}, {len(plan.scenes)} scenes, "
            f"{len(plan.variants)} clips, 0 calib"
        )
    ]
