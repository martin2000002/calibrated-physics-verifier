"""The gates command refuses before reading any sample or computing any score: a sealed training
selection, a background selection other than the training one, a training clip off split=train,
a sample file of another arm or naming clips outside its selection, an artifact of another check.
A well-formed run scores the internal validation the artifact was trained with."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from records import manifest

from physverify.app import learned, learned_gates
from physverify.app.learned_gates import (
    BackgroundSelectionError,
    GateFiles,
    GateSelections,
    SampleFileError,
    gate_report,
    plan_gates,
    read_samples,
    scene_background,
    target_zones,
)
from physverify.calibration.sealed_ledger import SealedSetError
from physverify.checks.variants._learned import LearnedArtifact
from physverify.core.errors import UsageError
from physverify.extraction.pair_features import ZONE_MAX
from physverify.learned.arms import PERCEIVED_ARM, TRUTH_ARM
from physverify.learned.image_table import ALL_FRAMES, MIDDLE_FRAME
from physverify.learned.samples import Role, Sample, SampleSet, TableSource, clip_role
from physverify.learned.train import TrainSettings, is_validation
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.scene import MagnitudeKind, PerturbationRecord

TRAIN = ("kubric_synthetic", "train_h")
HDRI = ("kubric_synthetic", "hdri_dev")
MISSING = Path("does-not-exist.npz")
FILES = GateFiles(MISSING, MISSING, MISSING, MISSING)


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


def test_a_background_selection_other_than_the_training_one_is_refused_before_any_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unread(app: object, name: str) -> object:
        raise AssertionError("the dataset was read")

    monkeypatch.setattr(learned, "dataset_for", unread)
    with pytest.raises(BackgroundSelectionError, match="hdri_dev"):
        report(HDRI)
    with pytest.raises(BackgroundSelectionError):
        plan_gates(None, GateSelections(TRAIN, HDRI), "R1", err=print)  # type: ignore[arg-type]


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


def test_a_scene_background_is_its_rendered_hdri_its_kind_or_unknown() -> None:
    hdri = SimpleNamespace(background="hdri", seed=3)
    assert scene_background(hdri, ("a", "b")) == "b"  # type: ignore[arg-type]
    assert scene_background(SimpleNamespace(background="plain"), ()) == "plain"  # type: ignore[arg-type]
    assert scene_background(None, ("a",)) == "unknown"


SHARE = 0.35
"""Not the training default, so the report must read it from the artifact."""
SHA = "e" * 64


def perturbation(name: str, checks: tuple[str, ...]) -> PerturbationRecord:
    return PerturbationRecord(
        name, checks, 0.2, "pen_over_size", MagnitudeKind.CHECK_UNIT, 1, ("a", "b"), None, 7
    )


ROLES = {
    Role.CLEAN: None,
    Role.POSITIVE: perturbation("r1", ("R1",)),
    Role.DECOY: perturbation("r1_decoy", ()),
}


def pick(n: int, keep: object) -> list[str]:
    found = (f"s{i}" for i in range(10_000))
    return [s for s in found if keep(s)][:n]  # type: ignore[operator]


def scene_clips() -> tuple[list[str], list[str], tuple[ClipManifest, ...]]:
    """Validation scenes under the artifact's share, half of them outside the default one."""
    default = TrainSettings().validation_share
    held = pick(12, lambda s: is_validation(s, default)) + pick(
        12, lambda s: is_validation(s, SHARE) and not is_validation(s, default)
    )
    trained = pick(6, lambda s: not is_validation(s, SHARE))
    clips = tuple(
        replace(
            manifest(),
            clip_id=f"{scene}_{role.value}",
            scene_id=scene,
            split=Split.TRAIN,
            perturbation=record,
        )
        for scene in held + trained
        for role, record in ROLES.items()
    )
    return held, trained, clips


def sample_file(path: Path, arm: str, rows: list[Sample], frames: str) -> Path:
    SampleSet(tuple(rows), ("d0",), ("token_000",), TableSource(frames, SHA)).to_table().save(path)
    return path


def test_a_well_formed_run_scores_the_validation_of_the_artifacts_share(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    held, trained, clips = scene_clips()
    seeds = {s: 2 * i + (i % 2 if s in held else 0) for i, s in enumerate(held + trained)}
    dataset = SimpleNamespace(clips=lambda subset: clips)
    monkeypatch.setattr(learned, "dataset_for", lambda app, name: dataset)
    monkeypatch.setattr(learned, "sealed_looks", lambda app, chosen, err: ())
    monkeypatch.setattr(learned_gates, "aimed", lambda app: {})
    monkeypatch.setattr(
        learned_gates,
        "load_dataset_settings",
        lambda settings: SimpleNamespace(effective_hdri_ids=lambda: ("seen", "unseen")),
    )
    monkeypatch.setattr(
        learned_gates,
        "_scene",
        lambda dataset, clip: SimpleNamespace(
            background="hdri", seed=seeds[clip.scene_id], camera=None
        ),
    )
    artifact = SimpleNamespace(
        check_id="R1",
        encoder_sha256=SHA,
        validation_share=SHARE,
        estimate=lambda dense, tokens: dense[:, 0],
    )
    monkeypatch.setattr(LearnedArtifact, "load", staticmethod(lambda path: artifact))

    def rows(arm: str) -> list[Sample]:
        level = {s: i / 100 for i, s in enumerate(held)}
        return [
            Sample(
                c.clip_id,
                c.scene_id or "",
                ("a",),
                arm,
                role,
                1.0 if role is Role.POSITIVE else 0.0,
                np.array([level[c.scene_id or ""] + (10.0 if role is Role.POSITIVE else 0.0)]),
                np.zeros(1),
            )
            for c in clips
            if c.scene_id in level
            for role in [clip_role(c.perturbation, "R1")]
            if role is not None
        ]

    files = GateFiles(
        MISSING,
        sample_file(tmp_path / "t1.npz", TRUTH_ARM, rows(TRUTH_ARM), ALL_FRAMES),
        sample_file(tmp_path / "t0.npz", PERCEIVED_ARM, rows(PERCEIVED_ARM), ALL_FRAMES),
        sample_file(tmp_path / "s.npz", PERCEIVED_ARM, rows(PERCEIVED_ARM), MIDDLE_FRAME),
    )
    monkeypatch.setattr(learned_gates, "file_sha256", lambda path: "x")
    app = SimpleNamespace(settings=None)
    found = gate_report(app, GateSelections(TRAIN, TRAIN), "R1", files, err=print)  # type: ignore[arg-type]
    statuses = {name: gate["status"] for name, gate in found["gates"].items()}  # type: ignore[attr-defined]
    assert statuses == {
        "G-L0": "pass",
        "G-L1": "pass",
        "C-dec": "pass",
        "C-inv": "not_evaluable",
        "C-scr": "pass",
        "C-hdri": "pass",
    }
    assert found["verdict"] == {"G-L0": True, "G-L1_with_controls": True, "C-hdri_flagged": False}
    assert found["validation_share"] == SHARE
    populations = found["populations"]
    assert populations["validation"] == {"clean": 24, "positive": 24, "decoy": 24}  # type: ignore[index]
    assert populations["C-hdri"] == {  # type: ignore[index]
        "positive_seen": 12,
        "clean_seen": 12,
        "positive_unseen": 12,
        "clean_unseen": 12,
    }
    unscored = found["unscored"]
    assert all(s["unscored"] == 0 for g in unscored.values() for s in g.values())  # type: ignore[attr-defined]
