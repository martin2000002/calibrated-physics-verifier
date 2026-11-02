"""The identity of a dataset's content enters every stage key, ground truth included, so two
versions with the same name and clip ids never share cache entries."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from fakes import FAKE
from settings_factory import SettingsFactory
from support import chain_keys, plan_overrides, run_pipeline
from synthetic_clip import GT_VERSION, SyntheticClip, SyntheticFixtureDataset
from wiring import resolve_plan

from physverify.app.plugins import PLUGIN_KINDS
from physverify.config.models import Settings
from physverify.datasets._base import CacheIdentitySource, DatasetSource, dataset_identity
from physverify.datasets._kubric_synthetic import KubricSynthetic, content_identity
from physverify.datasets.kubric_fresh import KubricFresh
from physverify.extraction._base import StageProvider
from physverify.extraction.pipeline import ExtractionReport
from physverify.schema.stages import STAGE_ORDER, Stage
from physverify.synth.generate import save_plan
from physverify.synth.plan import DatasetPlan
from physverify.synth.settings import load_dataset_settings
from physverify.synth.store import DatasetStore

Providers = Mapping[str, type[StageProvider]]
CALIB, HDRI_DEV = "v1-precise", "v1-precise-hdri-dev"
CELLS = {
    "ground_truth": ((), STAGE_ORDER),
    "perceived": (plan_overrides(detect=FAKE, track=FAKE, depth=FAKE), (Stage.CAMERA,)),
}
NO_IDENTITY = ("physics_iq", "physics_iq_views", "pisa_real", "objectron_static")


class VersionedFixture(SyntheticFixtureDataset):
    """The fixture clip under a dataset version: same name and clip ids, its own identity."""

    def __init__(self, version: str) -> None:
        super().__init__()
        self._version = version

    def cache_identity(self) -> str:
        return content_identity(self._version, "fixture")


def cached(report: ExtractionReport) -> set[bool]:
    return {event.cached for event in report.events}


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_another_version_misses_every_stage_and_the_first_still_hits(
    make_settings: SettingsFactory, providers: Providers, cell: str
) -> None:
    overrides, gt = CELLS[cell]
    settings = make_settings(*overrides)
    first = run_pipeline(settings, providers, VersionedFixture(CALIB), gt_stages=gt)
    other = run_pipeline(settings, providers, VersionedFixture(HDRI_DEV), gt_stages=gt)
    again = run_pipeline(settings, providers, VersionedFixture(CALIB), gt_stages=gt)
    assert {event.stage for event in other.events} == set(STAGE_ORDER)
    assert cached(first) == cached(other) == {False}
    assert cached(again) == {True}
    assert dict(again.keys) == dict(first.keys)
    assert all(other.keys[cell_key] != key for cell_key, key in first.keys.items())


def test_a_dataset_without_identity_keeps_its_keys(
    make_settings: SettingsFactory, providers: Providers, fixture_clip: SyntheticClip
) -> None:
    settings = make_settings()
    dataset = SyntheticFixtureDataset()
    report = run_pipeline(settings, providers, dataset, gt_stages=STAGE_ORDER)
    plan = resolve_plan(settings, providers, dataset=dataset.name, gt_stages=STAGE_ORDER)
    former = chain_keys(plan, fixture_clip.manifest, gt_version=GT_VERSION, profile=None)
    assert dataset_identity(dataset) is None
    assert {stage: report.keys[(fixture_clip.manifest.clip_id, stage)] for stage in former} == (
        former
    )


@pytest.mark.parametrize("name", NO_IDENTITY)
def test_real_datasets_declare_no_identity(name: str) -> None:
    assert not issubclass(PLUGIN_KINDS["datasets"]()[name], CacheIdentitySource)


def test_synthetic_versions_key_apart(
    make_settings: SettingsFactory, providers: Providers, fixture_clip: SyntheticClip
) -> None:
    settings = make_settings()
    calib, hdri_dev = (synthetic(settings, version) for version in (CALIB, HDRI_DEV))
    assert dataset_identity(calib) != dataset_identity(hdri_dev)
    assert_keys_apart(settings, providers, fixture_clip, calib, hdri_dev)


def test_fresh_versions_key_apart(
    make_settings: SettingsFactory, providers: Providers, fixture_clip: SyntheticClip
) -> None:
    fresh = [
        fresh_of(make_settings(f"kubric_fresh.version={version}"), version)
        for version in ("v1-fresh", "v1-fresh2")
    ]
    assert_keys_apart(make_settings(), providers, fixture_clip, *fresh)


def test_the_identity_is_the_stored_plans_not_the_settings(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    declared = load_dataset_settings(settings)
    stored = synthetic(settings, HDRI_DEV, content_hash="rendered-elsewhere")
    assert stored.cache_identity() == content_identity(HDRI_DEV, "rendered-elsewhere")
    assert stored.cache_identity() != content_identity(declared.version, declared.content_hash())


def planned(settings: Settings, version: str, content_hash: str | None = None) -> DatasetStore:
    """An empty store of ``version`` whose plan records ``content_hash`` (the declared one)."""
    store = DatasetStore(settings.paths.synthetic_dir / version)
    digest = content_hash or load_dataset_settings(settings).content_hash()
    save_plan(store, DatasetPlan(version, digest, (), (), {}))
    return store


def synthetic(settings: Settings, version: str, content_hash: str | None = None) -> KubricSynthetic:
    return KubricSynthetic(settings, root=planned(settings, version, content_hash).root)


def fresh_of(settings: Settings, version: str) -> KubricFresh:
    planned(settings, version)
    return KubricFresh(settings)


def assert_keys_apart(
    settings: Settings,
    providers: Providers,
    clip: SyntheticClip,
    one: DatasetSource,
    other: DatasetSource,
) -> None:
    """Every stage key, ground truth included, differs between the two datasets' identities."""
    plan = resolve_plan(settings, providers, dataset=one.name, gt_stages=STAGE_ORDER)
    keys = [
        chain_keys(plan, clip.manifest, gt_version=GT_VERSION, profile=None, dataset=identity)
        for identity in (dataset_identity(one), dataset_identity(other))
    ]
    assert set(keys[0]) == set(STAGE_ORDER)
    assert all(keys[0][stage] != keys[1][stage] for stage in STAGE_ORDER)
