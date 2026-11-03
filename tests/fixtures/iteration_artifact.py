"""The calibration.json of a run_perception_iteration.sh directory, as the freeze reads it."""

from __future__ import annotations

from physverify.calibration.artifact import (
    CalibrationArtifact,
    DatasetRef,
    SettingsSnapshot,
    StageSigma,
)
from physverify.schema.io import stable_hash

POPULATION = ("clip-a", "clip-b", "clip-c")


def calibration_text(name: str, *, whole: bool = True) -> str:
    """A calibration.json of iteration ``name`` fitted on its whole calib population, or on one
    chosen clip of it (a rehearsal)."""
    clips = POPULATION if whole else POPULATION[:1]
    dataset = DatasetRef(
        name="kubric_synthetic",
        version="v1",
        subset="full",
        split="calib",
        n_clips=len(clips),
        n_scenes=len(clips),
        clip_ids_hash=stable_hash(sorted(clips)),
        population_hash=stable_hash(sorted(POPULATION)),
    )
    snapshot = SettingsSnapshot(
        alpha=0.05, beta=0.05, k_sigma=2.0, folds=5, bootstrap=200, seed=0, ci=0.95,
        min_sigma_pairs=10, platt_l2=0.01,
    )  # fmt: skip
    return CalibrationArtifact(
        status="insufficient",
        created_utc="2026-11-03T00:00:00Z",
        run_id=f"run of {name}",
        commit="c" * 40,
        config_hash="cfg",
        checks_hash="checks",
        error_profile_hash="profile",
        dataset=dataset,
        settings=snapshot,
        strata=(),
        stage_sigma=StageSigma(path=None, hash=None),
    ).to_json()
