"""Hand-made calibration artifacts shared by the A8 tests and by its consumers (H2, H3): one
stratum builder, so every test of the stratum gate sees the same artifact and messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from physverify.calibration.loader import Stratum as StratumView
from physverify.calibration.schema import (
    Aggregate,
    ArtifactStatus,
    Auroc,
    CalibrationArtifact,
    CheckCalibration,
    CheckStatus,
    Ci,
    DatasetRef,
    DetectionLimit,
    Fpr,
    Monotonicity,
    Probability,
    Rate,
    SettingsSnapshot,
    SigmaCorrection,
    StageSigma,
    Stratum,
    Weight,
)
from physverify.schema.checks import InputLevel

DATASET = "calib_fixture"
PROFILE = "fast"
FIXTURE_CHECKS_HASH = "checks"
"""``checks_hash`` the hand-made artifacts are stamped with."""


def dataset_ref(n_clips: int = 0) -> DatasetRef:
    return DatasetRef(
        name=DATASET,
        version="v1",
        subset="test",
        split="calib",
        n_clips=n_clips,
        n_scenes=0,
        clip_ids_hash="hash",
    )


def _rate() -> Rate:
    return Rate(value=None, ci=Ci(), k=0, n=0)


def check_calibration(
    *,
    tau: float | None,
    a: float | None = 0.0,
    b: float | None = 4.0,
    scale: float = 0.1,
    weight: float = 1.0,
    bias: float = 0.0,
    factor: float | None = 1.0,
    status: CheckStatus = "ok",
    unit: str = "u",
) -> CheckCalibration:
    empty = Auroc(value=None, ci=Ci(), n_positive=0, n_negative=0)
    return CheckCalibration(
        unit=unit,
        status=status,
        tau=tau,
        n_clean_scenes=20,
        n_positive_clips=20,
        n_off_target_clips=0,
        fpr=Fpr(
            cv=_rate(),
            cv_decided=_rate(),
            in_sample=_rate(),
            in_sample_decided=_rate(),
            off_target=_rate(),
            undeclared_coupling=False,
        ),
        tpr_by_level=(),
        auroc=empty,
        abstention_rate=_rate(),
        detection_limit=DetectionLimit(
            value=None, ci=Ci(), censored="none", bound=None, unit=unit, beta=0.05, n=0
        ),
        monotonicity=Monotonicity(
            kendall_tau_b=None,
            ci=Ci(),
            n_pairs=0,
            n_scenes=0,
            median_delta_by_level={},
            monotone_medians=True,
            h1_pass=False,
        ),
        weight=Weight(
            value=weight, informedness=None, informedness_ci=Ci(), n_positive=0, n_negative=0
        ),
        probability=Probability(a=a, b=b, scale=scale, ece_cv=None, ece_ci=Ci(), n=0),
        sigma=SigmaCorrection(
            bias=bias,
            bias_ci=Ci(),
            factor=factor,
            factor_ci=Ci(),
            n_pairs=0,
            coverage_before=(),
            coverage_after=(),
        ),
    )


def stratum_model(
    checks: Mapping[str, CheckCalibration],
    level: InputLevel = InputLevel.PERCEIVED,
    score_threshold: float | None = None,
    profile: str = PROFILE,
) -> Stratum:
    empty = Auroc(value=None, ci=Ci(), n_positive=0, n_negative=0)
    return Stratum(
        level=level,
        profile=profile,
        row_step=1,
        k_sigma=2.0,
        on_unknown_sigma="abstain",
        checks=dict(checks),
        aggregate=Aggregate(
            score_threshold=score_threshold,
            sufficient=score_threshold is not None,
            n_clean_scenes=0,
            n_perturbed_clips=0,
            auroc=empty,
        ),
    )


def view(checks: Mapping[str, CheckCalibration]) -> StratumView:
    return StratumView(model=stratum_model(checks), source="test", artifact=None)


def artifact_of(
    strata: Sequence[Stratum],
    run_id: str = "run-a",
    *,
    status: ArtifactStatus = "insufficient",
    checks_hash: str = FIXTURE_CHECKS_HASH,
) -> CalibrationArtifact:
    return CalibrationArtifact(
        status=status,
        created_utc="2026-10-17T00:00:00Z",
        run_id=run_id,
        commit="abc",
        config_hash="cfg",
        checks_hash=checks_hash,
        error_profile_hash="profile",
        dataset=dataset_ref(),
        settings=SettingsSnapshot(
            alpha=0.05,
            beta=0.05,
            k_sigma=2.0,
            folds=5,
            bootstrap=200,
            seed=0,
            ci=0.95,
            min_sigma_pairs=10,
            platt_l2=0.01,
        ),
        strata=tuple(strata),
        stage_sigma=StageSigma(path=None, hash=None),
    )


def write_artifact(
    path: Path,
    *,
    ready: bool = True,
    status: ArtifactStatus = "insufficient",
    profile: str = PROFILE,
    checks_hash: str = FIXTURE_CHECKS_HASH,
) -> Path:
    """A perceived stratum of ``profile`` with R1 ok (tau 1); ``ready`` = a sufficient aggregate,
    otherwise the stratum loads but cannot score (``UnusableStratumError``)."""
    checks = {
        "R1": check_calibration(tau=1.0),
        "R5b": check_calibration(tau=None, status="no_negatives"),
    }
    cut = 0.5 if ready else None
    stratum = stratum_model(checks, InputLevel.PERCEIVED, cut, profile)
    artifact = artifact_of((stratum,), status=status, checks_hash=checks_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.to_json(), encoding="utf-8")
    return path
