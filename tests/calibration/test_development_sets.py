"""Development datasets: declared by name and version, read as calib by the calib-only commands."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pytest
from calibration_cases import DATASET, settings, synthetic_report

from physverify.calibration.development import (
    calib_clips,
    development_clips,
    is_development,
    may_be_development,
)
from physverify.core.errors import UsageError
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.datasets._kubric_synthetic import content_identity
from physverify.schema.clip import ClipManifest, Split

CLIPS = synthetic_report(3, split=Split.EVAL, prefix="d").clips
DECLARED = settings(development_sets=[f"{DATASET}@v2"])


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
