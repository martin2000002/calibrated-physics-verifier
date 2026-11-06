"""The gates command refuses before reading any sample or computing any score: a sealed training or
unseen-background selection, a training clip off split=train, a sample file of another arm or
naming clips outside its selection, an artifact of another check."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from records import manifest

from physverify.app import learned
from physverify.app.learned_gates import (
    GateFiles,
    GateSelections,
    SampleFileError,
    gate_report,
    plan_gates,
    read_samples,
    target_zones,
)
from physverify.calibration.sealed_ledger import SealedSetError
from physverify.checks.variants._learned import LearnedArtifact
from physverify.core.errors import UsageError
from physverify.extraction.pair_features import ZONE_MAX
from physverify.learned.arms import PERCEIVED_ARM, TRUTH_ARM
from physverify.learned.image_table import ALL_FRAMES, MIDDLE_FRAME
from physverify.learned.samples import Role, Sample, SampleSet, TableSource
from physverify.schema.clip import Split

TRAIN = ("kubric_synthetic", "train_h")
HDRI = ("kubric_synthetic", "hdri_dev")
MISSING = Path("does-not-exist.npz")
FILES = GateFiles(MISSING, MISSING, MISSING, MISSING, MISSING)


def selections(
    monkeypatch: pytest.MonkeyPatch, sealed: tuple[str, ...], split: Split = Split.TRAIN
) -> None:
    clips = (replace(manifest(), split=split),)
    dataset = SimpleNamespace(clips=lambda subset: clips)
    monkeypatch.setattr(learned, "dataset_for", lambda app, name: dataset)
    monkeypatch.setattr(
        learned,
        "sealed_looks",
        lambda app, chosen, err: tuple(s.subset for s in chosen if s.subset in sealed),
    )


def report(hdri: tuple[str, str | None] | None = None) -> object:
    return gate_report(None, GateSelections(TRAIN, hdri), "R1", FILES, err=print)  # type: ignore[arg-type]


def test_a_sealed_training_selection_is_refused_before_any_file_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selections(monkeypatch, ("train_h",))
    with pytest.raises(SealedSetError, match="nothing was read"):
        report()
    with pytest.raises(SealedSetError):
        plan_gates(None, GateSelections(TRAIN), "R1", err=print)  # type: ignore[arg-type]


def test_a_sealed_unseen_background_selection_is_refused_before_any_file_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selections(monkeypatch, ("hdri_dev",))
    with pytest.raises(SealedSetError, match="hdri_dev"):
        report(HDRI)


def test_a_training_clip_off_split_train_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    selections(monkeypatch, (), Split.CALIB)
    with pytest.raises(UsageError, match="not split=train"):
        report()


SOURCE = TableSource(ALL_FRAMES, "e" * 64)


def samples(tmp_path: Path, arm: str, clip_id: str = "c0", source: TableSource = SOURCE) -> Path:
    one = Sample(clip_id, "s0", ("a",), arm, Role.POSITIVE, 1.0, np.zeros(1), np.zeros(1))
    path = tmp_path / "samples.npz"
    SampleSet((one,), ("d0",), ("token_000",), source).to_table().save(path)
    return path


def test_a_sample_file_of_another_arm_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SampleFileError, match="reads A_fff"):
        read_samples(samples(tmp_path, TRUTH_ARM), "A_fff", frozenset({"c0"}), SOURCE)


def test_a_sample_file_naming_clips_outside_its_selection_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SampleFileError, match="outside the selection"):
        read_samples(
            samples(tmp_path, PERCEIVED_ARM, "elsewhere"), PERCEIVED_ARM, frozenset(), SOURCE
        )


def test_a_sample_file_of_its_arm_and_selection_is_read(tmp_path: Path) -> None:
    found = read_samples(samples(tmp_path, PERCEIVED_ARM), PERCEIVED_ARM, frozenset({"c0"}), SOURCE)
    assert [s.clip_id for s in found.samples] == ["c0"]
    assert found.source == SOURCE


@pytest.mark.parametrize(
    "written",
    [
        TableSource(MIDDLE_FRAME, SOURCE.encoder_sha256),
        TableSource(ALL_FRAMES, "f" * 64),
        TableSource(),
    ],
)
def test_a_sample_file_read_from_other_frames_or_another_encoder_is_refused(
    tmp_path: Path, written: TableSource
) -> None:
    path = samples(tmp_path, PERCEIVED_ARM, source=written)
    with pytest.raises(SampleFileError, match="rows read from"):
        read_samples(path, PERCEIVED_ARM, frozenset({"c0"}), SOURCE)


def test_an_artifact_of_another_check_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    selections(monkeypatch, ())
    monkeypatch.setattr(
        LearnedArtifact,
        "load",
        staticmethod(lambda path: SimpleNamespace(check_id="R2")),
    )
    with pytest.raises(SampleFileError, match="an artifact of R2"):
        report()


def test_the_target_zone_is_the_largest_over_target_subjects_only() -> None:
    def one(clip: str, label: float, zone: float) -> Sample:
        return Sample(
            clip, clip, ("a",), TRUTH_ARM, Role.POSITIVE, label, np.array([zone]), np.zeros(0)
        )

    rows = (one("c", 1.0, 0.0), one("c", 1.0, 0.2), one("c", 0.0, 0.9), one("d", 0.0, 0.5))
    assert target_zones(SampleSet(rows, (ZONE_MAX,), ())) == {"c": pytest.approx(0.2)}
    assert target_zones(SampleSet(rows, ("other",), ())) == {}
    assert target_zones(None) == {}
