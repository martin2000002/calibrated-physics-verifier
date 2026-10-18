"""Builders shared by the calibration tests: rows, manifests, a synthetic raw report, artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from physverify.calibration.collect import level_data
from physverify.calibration.fit import FitInputs, FitMeta, LevelData, fit_calibration
from physverify.calibration.loader import Stratum as StratumView
from physverify.calibration.schema import (
    Aggregate,
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
from physverify.calibration.settings import CalibrationConfig, CalibrationSettings
from physverify.calibration.table import CalibrationRow, Role
from physverify.schema.checks import CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest, Split, TimeSource
from physverify.schema.scene import MagnitudeKind, PerturbationRecord
from physverify.verification.policies.threshold import ThresholdParams
from physverify.verification.runner import RowStatus, TableRow, VerificationReport
from physverify.verification.sweep import CheckUnit

DATASET = "calib_fixture"
PROFILE = "fast"
CHECKS = (
    CheckUnit("R1", "pen_over_size"),
    CheckUnit("R3a", "e_minus_1"),
    CheckUnit("R5b", "g_rel_error"),
)
UNITS = {c.check_id: c.magnitude_unit for c in CHECKS}
PERTURBATIONS = {"r1_penetration": "R1", "r5b_gravity_scale": "R5b"}
LEVELS = (1, 2, 3)
CLEAN_NOISE = {"R1": 0.01, "R3a": 0.02}
"""R5b has no event on clean clips (no_negatives); no perturbation targets R3a (no_positives)."""


def settings(**update: object) -> CalibrationSettings:
    return CalibrationSettings.model_validate({"bootstrap": 200, **update})


def config(**update: object) -> CalibrationConfig:
    return CalibrationConfig(
        calibration=settings(**update),
        policy=ThresholdParams(),
        couplings=frozenset({("r5a_gravity_switch", "R5b")}),
        min_span_iou=0.5,
    )


def record(
    name: str,
    targets: tuple[str, ...],
    magnitude: float,
    unit: str,
    level: int = 1,
    kind: MagnitudeKind = MagnitudeKind.CHECK_UNIT,
) -> PerturbationRecord:
    return PerturbationRecord(
        name=name,
        target_checks=targets,
        magnitude=magnitude,
        magnitude_unit=unit,
        magnitude_kind=kind,
        level=level,
        target_objects=("a",),
        frame_span=None,
        seed=0,
    )


def clip(
    clip_id: str,
    scene_id: str | None = None,
    split: Split = Split.CALIB,
    perturbation: PerturbationRecord | None = None,
) -> ClipManifest:
    return ClipManifest(
        dataset=DATASET,
        clip_id=clip_id,
        split=split,
        video_path=f"{DATASET}/{clip_id}",
        n_frames=24,
        width=64,
        height=64,
        container_fps=24.0,
        seconds_per_frame=1.0 / 24.0,
        time_source=TimeSource.GROUND_TRUTH,
        scene_id=scene_id,
        perturbation=perturbation,
    )


def result(
    check_id: str,
    clip_id: str = "clip",
    magnitude: float | None = 0.1,
    sigma: float = 0.0,
    level: InputLevel = InputLevel.PERCEIVED,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        clip_id=clip_id,
        subject=("a",),
        frame_span=(0, 3),
        magnitude=magnitude,
        sigma=sigma,
        unit=UNITS.get(check_id, "u"),
        verdict=Verdict.UNSCORED,
        input_level=level,
    )


def row(
    magnitude: float | None,
    *,
    role: Role = Role.CLEAN,
    scene: str = "s0",
    clip_id: str | None = None,
    level: int = 0,
    mu: float | None = None,
    sigma: float = 0.0,
    check_id: str = "R1",
    status: RowStatus = RowStatus.SCORED,
) -> CalibrationRow:
    return CalibrationRow(
        clip_id=clip_id or f"{scene}-{role.value}-{level}",
        scene_id=scene,
        split=Split.CALIB,
        check_id=check_id,
        unit=UNITS.get(check_id, "u"),
        level=InputLevel.ORACLE,
        status=status,
        magnitude=magnitude,
        sigma=sigma,
        role=role,
        perturbation="none" if role is Role.CLEAN else "pert",
        pert_level=level,
        mu=mu,
        mu_unit=UNITS.get(check_id, "u"),
        row_step=1,
    )


# --- a synthetic raw-policy report ----------------------------------------------------------------


def _magnitudes(rng: np.random.Generator, target: str | None, mu: float) -> dict[str, float | None]:
    found: dict[str, float | None] = {
        c: abs(float(rng.normal(0.0, s))) for c, s in CLEAN_NOISE.items()
    }
    found["R5b"] = None
    if target is not None:
        found[target] = mu + float(rng.normal(0.0, 0.005))
    return found


def _rows(
    manifest: ClipManifest, magnitudes: Mapping[str, float | None], sigma: float, lv: InputLevel
) -> tuple[list[TableRow], list[CheckResult]]:
    table: list[TableRow] = []
    results: list[CheckResult] = []
    for check in CHECKS:
        m = magnitudes.get(check.check_id)
        status = RowStatus.NO_EVENT if m is None else RowStatus.SCORED
        table.append(
            TableRow(
                clip_id=manifest.clip_id,
                check_id=check.check_id,
                status=status,
                magnitude=m,
                sigma=math.nan if m is None else sigma,
                verdict=None if m is None else Verdict.UNSCORED,
                abstain_reason=None,
                n_subjects=0 if m is None else 1,
                n_abstained=0,
                unit=check.magnitude_unit,
                row_step=1,
            )
        )
        if m is not None:
            results.append(result(check.check_id, manifest.clip_id, m, sigma, lv))
    return table, results


def scene_clips(scene: str, split: Split = Split.CALIB) -> list[tuple[ClipManifest, str | None]]:
    found: list[tuple[ClipManifest, str | None]] = [(clip(f"{scene}__none", scene, split), None)]
    for name, target in PERTURBATIONS.items():
        for lv in LEVELS:
            mu = 0.05 * 2 ** (lv - 1)
            pert = record(name, (target,), mu, UNITS[target], lv)
            found.append((clip(f"{scene}__{name}_L{lv}", scene, split, pert), target))
    return found


def synthetic_report(
    n_scenes: int = 40,
    seed: int = 0,
    *,
    split: Split = Split.CALIB,
    level: InputLevel = InputLevel.ORACLE,
    sigma: float = 0.0,
    prefix: str = "s",
) -> VerificationReport:
    """1 clean + 2 perturbations x 3 levels per scene: R1 ok, R3a no_positives, R5b no_negatives."""
    rng = np.random.default_rng(seed)
    table: list[TableRow] = []
    results: list[CheckResult] = []
    clips: list[ClipManifest] = []
    for s in range(n_scenes):
        for manifest, target in scene_clips(f"{prefix}{s:03d}", split):
            mu = 0.0 if manifest.perturbation is None else manifest.perturbation.magnitude
            rows, found = _rows(manifest, _magnitudes(rng, target, mu), sigma, level)
            table += rows
            results += found
            clips.append(manifest)
    return VerificationReport(results=tuple(results), table=tuple(table), clips=tuple(clips))


def synthetic_level(report: VerificationReport, level: InputLevel) -> LevelData:
    return level_data(level, report, CHECKS, config(), split=Split.CALIB)


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


def fit_meta(run_id: str = "run-a", created_utc: str = "2026-10-17T00:00:00Z") -> FitMeta:
    return FitMeta(
        created_utc=created_utc,
        run_id=run_id,
        commit="abc",
        config_hash="cfg",
        checks_hash="checks",
        error_profile_hash="profile",
        profile=PROFILE,
        dataset=dataset_ref(),
        stage_sigma=StageSigma(path=None, hash=None),
    )


def fitted_artifact(
    n_scenes: int = 40, seed: int = 0, run_id: str = "run-a", **update: object
) -> CalibrationArtifact:
    report = synthetic_report(n_scenes, seed)
    inputs = FitInputs(
        (synthetic_level(report, InputLevel.ORACLE),), CHECKS, fit_meta(run_id=run_id)
    )
    return fit_calibration(inputs, config(**update))


# --- hand-made strata for apply -------------------------------------------------------------------


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
) -> Stratum:
    empty = Auroc(value=None, ci=Ci(), n_positive=0, n_negative=0)
    return Stratum(
        level=level,
        profile=PROFILE,
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


def artifact_of(strata: Sequence[Stratum], run_id: str = "run-a") -> CalibrationArtifact:
    return CalibrationArtifact(
        status="insufficient",
        created_utc="2026-10-17T00:00:00Z",
        run_id=run_id,
        commit="abc",
        config_hash="cfg",
        checks_hash="checks",
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
