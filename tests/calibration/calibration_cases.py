"""Builders shared by the calibration tests: rows, manifests, a synthetic raw report, artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from stratum_cases import (
    DATASET,
    PROFILE,
    artifact_of,
    check_calibration,
    dataset_ref,
    stratum_model,
    view,
)

from physverify.calibration.collect import level_data
from physverify.calibration.fit import FitInputs, FitMeta, LevelData, fit_calibration
from physverify.calibration.schema import (
    CalibrationArtifact,
    StageSigma,
)
from physverify.calibration.settings import CalibrationConfig, CalibrationSettings
from physverify.calibration.table import CalibrationRow, Role
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.schema.clip import ClipManifest, Split, TimeSource
from physverify.schema.scene import MagnitudeKind, PerturbationRecord
from physverify.verification.policies.threshold import ThresholdParams
from physverify.verification.runner import RowStatus, TableRow, VerificationReport
from physverify.verification.sweep import CheckUnit

__all__ = ["DATASET", "PROFILE", "artifact_of", "check_calibration", "stratum_model", "view"]

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
    subject: str = "a",
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        clip_id=clip_id,
        subject=(subject,),
        frame_span=(0, 3),
        magnitude=magnitude,
        sigma=sigma,
        unit=UNITS.get(check_id, "u"),
        verdict=Verdict.UNSCORED,
        input_level=level,
    )


def missing(
    check_id: str,
    clip_id: str = "clip",
    level: InputLevel = InputLevel.PERCEIVED,
    subject: str = "b",
) -> CheckResult:
    """A subject the runner abstained on for a missing input (no magnitude)."""
    return CheckResult(
        check_id=check_id,
        clip_id=clip_id,
        subject=(subject,),
        frame_span=None,
        magnitude=None,
        sigma=float("nan"),
        unit=UNITS.get(check_id, "u"),
        verdict=Verdict.ABSTAIN,
        abstain_reason=AbstainReason.MISSING_INPUT,
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
    subjects: Sequence[CheckResult] | None = None,
) -> CalibrationRow:
    """A row; without ``subjects`` it has the one subject its magnitude and status imply."""
    clip_id = clip_id or f"{scene}-{role.value}-{level}"
    if subjects is None:
        subjects = _one_subject(clip_id, check_id, magnitude, sigma, status)
    return CalibrationRow(
        clip_id=clip_id,
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
        subjects=tuple(subjects),
    )


def _one_subject(
    clip_id: str, check_id: str, magnitude: float | None, sigma: float, status: RowStatus
) -> tuple[CheckResult, ...]:
    if status is RowStatus.NO_EVENT:
        return ()
    if status is RowStatus.ABSTAINED or magnitude is None:
        return (missing(check_id, clip_id, InputLevel.ORACLE),)
    return (result(check_id, clip_id, magnitude, sigma, InputLevel.ORACLE),)


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
