"""The fresh Kubric renders of the final looks: their dataset overrides and their stored plan."""

from __future__ import annotations

from collections.abc import Mapping

from physverify.config.models import Settings
from physverify.datasets._base import DatasetSource
from physverify.datasets.kubric_fresh import VERSION_ENV, KubricFresh, KubricFreshSettings
from physverify.freeze.errors import QueryError
from physverify.schema.clip import Split
from physverify.synth.generate import load_plan
from physverify.synth.settings import dataset_root, load_dataset_settings
from physverify.synth.store import DatasetStore

__all__ = ("VERSION_ENV", "plan_lines", "render_overrides")


def render_overrides(
    fresh: KubricFreshSettings,
    datasets: Mapping[str, type[DatasetSource]],
    dataset: str,
    seed: int,
) -> dict[str, str]:
    """The overrides of the fresh Kubric set ``dataset``, found in the dataset registry."""
    source = datasets.get(dataset)
    if source is None or not issubclass(source, KubricFresh):
        raise QueryError(f"{dataset} is not a fresh Kubric set")
    return source.render_overrides(fresh, seed)


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
