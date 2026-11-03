"""Development datasets: declared by name and version, read as calib by the calib-only commands."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

import pytest
from calibration_cases import DATASET, settings, synthetic_report

from physverify.calibration.development import (
    calib_clips,
    development_clips,
    is_development,
    may_be_development,
    require_sealed,
)
from physverify.calibration.sealed_ledger import SealedSetError
from physverify.calibration.settings import calibration_settings
from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.datasets._kubric_synthetic import content_identity
from physverify.datasets.kubric_fresh import (
    SECTION,
    KubricFresh,
    KubricFreshBlind,
    KubricFreshOod,
    fresh_settings,
)
from physverify.schema.clip import ClipManifest, Split

CLIPS = synthetic_report(3, split=Split.EVAL, prefix="d").clips
DECLARED = settings(development_sets=[f"{DATASET}@v2"])
SECTION_VALUES = {
    "blind_scenes": ["s001"],
    "scenes_per_family": 1,
    "ood_scenes_per_family": 1,
    "ood_hdri_pool": "pool",
}


class PlainDataset:
    name: ClassVar[str] = DATASET

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return CLIPS

    def video(self, clip: ClipManifest) -> VideoSource:
        raise NotImplementedError

    def ground_truth(self, clip: ClipManifest) -> GroundTruthStore | None:
        return None

    def fetch(self, subset: str | None, dest: Path) -> None:
        return None


class VersionedDataset(PlainDataset):
    def __init__(self, version: str) -> None:
        self._version = version

    def cache_identity(self) -> str:
        return content_identity(self._version, "fixture")

    def content_version(self) -> str:
        return self._version


class FreshRender(VersionedDataset):
    name: ClassVar[str] = "kubric_fresh"


def test_the_committed_sets_hold_both_fresh_renders(make_settings: Callable[..., Settings]) -> None:
    """Both kubric_fresh development renders are development sets; any other version of the name
    is not, and the sealed renders kubric_fresh reads by default are neither."""
    committed = make_settings()
    declared = calibration_settings(committed)
    assert declared.development_sets == (
        "kubric_fresh@v2-fresh",
        "kubric_fresh@v2-dev-b2",
        "kubric_synthetic@v1-precise-hdri-dev",
    )
    fresh = fresh_settings(committed)
    assert (fresh.version, fresh.ood_version) == ("v2-fresh2", "v2-fresh-ood")
    for version, expected in (("v2-fresh", True), ("v2-dev-b2", True), ("v2-fresh2", False)):
        assert is_development(FreshRender(version), declared) is expected
    for cls in (KubricFresh, KubricFreshOod, KubricFreshBlind):
        require_sealed(f"{cls.name}@{cls.sealed_version(committed.sections)}", declared)


def test_a_sealed_render_declared_a_development_set_is_refused() -> None:
    require_sealed(f"{DATASET}@v3", DECLARED)
    with pytest.raises(SealedSetError, match=f"{DATASET}@v2 is a sealed render and a declared"):
        require_sealed(f"{DATASET}@v2", DECLARED)


def test_the_sealed_fresh_versions_are_those_of_the_section() -> None:
    sections = {SECTION: {**SECTION_VALUES, "version": "a", "ood_version": "b"}}
    assert KubricFresh.sealed_version(sections) == "a"
    assert KubricFreshBlind.sealed_version(sections) == "a"
    assert KubricFreshOod.sealed_version(sections) == "b"


def test_a_development_set_names_its_version() -> None:
    with pytest.raises(ValueError, match="name@version"):
        settings(development_sets=[DATASET])


@pytest.mark.parametrize(
    ("source", "expected"),
    [(VersionedDataset("v2"), True), (VersionedDataset("v3"), False), (PlainDataset(), False)],
)
def test_only_the_declared_version_is_development(source: PlainDataset, expected: bool) -> None:
    assert is_development(source, DECLARED) is expected
    assert may_be_development(DATASET, DECLARED)
    assert not may_be_development("other", DECLARED)


def test_every_clip_of_a_development_set_is_calib() -> None:
    clips = calib_clips(VersionedDataset("v2"), "full", (), DECLARED)
    assert len(clips) == len(CLIPS) and {c.split for c in clips} == {Split.CALIB}


def test_another_version_keeps_its_splits() -> None:
    kept = development_clips(VersionedDataset("v3"), CLIPS, DECLARED)
    assert kept == list(CLIPS)


def test_another_version_has_no_calib_clip_to_fit() -> None:
    with pytest.raises(UsageError, match="no split=calib clip"):
        calib_clips(VersionedDataset("v3"), "full", (), DECLARED)
