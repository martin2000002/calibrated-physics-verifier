"""The calibration artifact ``calibration.json`` and the evaluation report (ADR 0009 §6).

Semver: a different major version is rejected; a newer minor version may only add optional
fields, which a lenient load drops. Every non-finite number is written as null.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field, ValidationInfo, model_validator

from physverify.diagnosis.report import ReportModel
from physverify.schema.checks import InputLevel

CALIBRATION_SCHEMA = "physverify.calibration"
CALIBRATION_VERSION = "1.2.0"
EVALUATION_SCHEMA = "physverify.calibration.evaluation"
LENIENT = "lenient"
"""Validation-context flag: drop keys this version does not know (newer minor version)."""

CheckStatus = Literal["ok", "insufficient", "no_positives", "no_negatives", "unranked_negatives"]
RateCi = Literal["clopper_pearson", "scene_bootstrap", "clopper_pearson_scenes"]
OpenSide = Literal["low", "high"]
ArtifactStatus = Literal["calibrated", "insufficient", "provisional"]
Censoring = Literal["none", "below", "above"]
OK: CheckStatus = "ok"
UNRANKED_NEGATIVES: CheckStatus = "unranked_negatives"
"""Clean clips have an event but every magnitude is NaN: the measurement is missing, not data."""


SCHEMA_KEY = "schema"
SCHEMA_FIELD = "schema_"


class ArtifactModel(ReportModel):
    @model_validator(mode="before")
    @classmethod
    def _drop_unknown(cls, data: Any, info: ValidationInfo) -> Any:
        if isinstance(data, dict) and info.context and info.context.get(LENIENT):
            known = {*cls.model_fields, *(f.alias for f in cls.model_fields.values() if f.alias)}
            return {key: value for key, value in data.items() if key in known}
        return data

    def to_json(self) -> str:
        payload = json.loads(super().to_json())
        if SCHEMA_FIELD in payload:
            payload[SCHEMA_KEY] = payload.pop(SCHEMA_FIELD)
        return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


class Ci(ArtifactModel):
    """Interval at the artifact's ``ci`` level; null bounds when it could not be computed."""

    low: float | None = None
    high: float | None = None
    non_finite: int | None = None
    """Bootstrap replicates whose statistic was not finite (counted, not silently dropped)."""


class Rate(ArtifactModel):
    """``k`` events out of ``n``. ``ci_method``: exact Clopper-Pearson over independent units;
    ``scene_bootstrap`` when several clips share a scene; ``clopper_pearson_scenes`` (exact over
    ``n_scenes``, conservative) when such a rate is 0 or 1 and the bootstrap would collapse."""

    value: float | None
    ci: Ci
    k: int
    n: int
    n_scenes: int | None = None
    ci_method: RateCi = "clopper_pearson"


class Fpr(ArtifactModel):
    """Totals count an abstention as no false positive; ``*_decided`` count over PASS/FAIL only."""

    cv: Rate
    """Scene-grouped cross-validation: tau fitted without the held-out clean scenes."""
    cv_decided: Rate
    in_sample: Rate
    in_sample_decided: Rate
    in_sample_optimistic: bool = True
    off_target: Rate
    """Perturbed clips not aimed at this check (couplings excluded); never fix tau. Several clips
    share a scene, so its interval is clustered by scene."""
    undeclared_coupling: bool
    off_target_failures: tuple[str, ...] = ()


class LevelTpr(ArtifactModel):
    level: int
    mu_median: float | None
    total: Rate
    """Abstentions count as misses."""
    decided: Rate


class Auroc(ArtifactModel):
    value: float | None
    ci: Ci
    n_positive: int
    n_negative: int
    in_sample: bool = True


class DetectionLimit(ArtifactModel):
    """L_D at miss rate ``beta`` in the injected unit. ``censored`` below: L_D <= ``bound``;
    above: L_D > ``bound``. A fitted value is an estimate. ``ci`` is the profile-likelihood
    interval; a side in ``ci_open`` did not close inside the tested range (its bound is null)."""

    value: float | None
    ci: Ci
    censored: Censoring
    bound: float | None
    unit: str
    beta: float
    n: int
    reason: str | None = None
    estimated: bool = True
    ci_method: str = "profile_likelihood"
    ci_open: tuple[OpenSide, ...] = ()


class Monotonicity(ArtifactModel):
    kendall_tau_b: float | None
    ci: Ci
    n_pairs: int
    n_scenes: int
    median_delta_by_level: dict[str, float | None]
    monotone_medians: bool
    h1_pass: bool


class Weight(ArtifactModel):
    value: float
    """max(0, informedness_ci.low) normalised by the largest weight of the stratum."""
    informedness: float | None
    informedness_ci: Ci
    n_positive: int
    n_negative: int


class Probability(ArtifactModel):
    """p = 1 / (1 + exp(-(a + b u))), u = (m - bias - tau) / scale; class-balanced Platt fit.
    An estimate conditional on the calib design, not a real-world probability."""

    a: float | None
    b: float | None
    scale: float
    ece_cv: float | None
    ece_ci: Ci
    n: int
    estimated: bool = True


class CoverageAt(ArtifactModel):
    k: float
    nominal: float
    observed: float | None
    n: int


class SigmaCorrection(ArtifactModel):
    """m' = m - bias, sigma' = factor * sigma; ``factor`` null = error not measured (abstain)."""

    bias: float
    bias_ci: Ci
    factor: float | None
    factor_ci: Ci
    n_pairs: int
    coverage_before: tuple[CoverageAt, ...]
    coverage_after: tuple[CoverageAt, ...]
    """Cross-fitted by scene."""
    reason: str | None = None
    estimated: bool = True


class CheckCalibration(ArtifactModel):
    unit: str
    status: CheckStatus
    tau: float | None
    tau_rule: str = "max(conformal_clean_scene, tolerance)"
    tau_tolerance: float = 0.0
    """Numerical tolerance epsilon_c (check units): tau = max(q, epsilon_c)."""
    tau_conformal: float | None = None
    """The conformal quantile q before the tolerance."""
    n_clean_scenes: int
    n_clean_dropped: int = 0
    """Clean scenes whose magnitude was NaN: counted, not ranked by the conformal quantile."""
    n_positive_clips: int
    n_off_target_clips: int
    fpr: Fpr
    tpr_by_level: tuple[LevelTpr, ...]
    auroc: Auroc
    abstention_rate: Rate
    detection_limit: DetectionLimit
    monotonicity: Monotonicity
    weight: Weight
    probability: Probability
    sigma: SigmaCorrection


class Aggregate(ArtifactModel):
    """A clip is flagged when 1 - S > ``score_threshold`` (conformal over clean scenes)."""

    score_threshold: float | None
    sufficient: bool
    score_threshold_in_sample: bool = True
    """s* is fitted on the clips that also fitted tau, p_c and the weights: optimistic."""
    n_clean_scenes: int
    n_perturbed_clips: int
    auroc: Auroc


class Stratum(ArtifactModel):
    level: InputLevel
    profile: str
    row_step: int
    k_sigma: float
    on_unknown_sigma: Literal["abstain", "point"]
    checks: dict[str, CheckCalibration]
    aggregate: Aggregate


class DatasetRef(ArtifactModel):
    name: str
    version: str
    subset: str
    split: str
    n_clips: int
    n_scenes: int
    clip_ids_hash: str


class SettingsSnapshot(ArtifactModel):
    alpha: float
    beta: float
    k_sigma: float
    folds: int
    bootstrap: int
    seed: int
    ci: float
    min_sigma_pairs: int
    platt_l2: float


class StageFactor(ArtifactModel):
    stage: str
    provider: str
    quantity: str
    factor: float | None
    n: int
    coverage_before: tuple[CoverageAt, ...]
    coverage_after: tuple[CoverageAt, ...]
    estimated: bool = True


class StageSigma(ArtifactModel):
    path: str | None
    hash: str | None
    entries: tuple[StageFactor, ...] = ()


class CalibrationArtifact(ArtifactModel):
    schema_: str = Field(default=CALIBRATION_SCHEMA, alias=SCHEMA_KEY)
    version: str = CALIBRATION_VERSION
    status: ArtifactStatus
    created_utc: str
    run_id: str
    commit: str
    config_hash: str
    checks_hash: str
    error_profile_hash: str
    dataset: DatasetRef
    settings: SettingsSnapshot
    strata: tuple[Stratum, ...]
    stage_sigma: StageSigma


class SceneKendall(ArtifactModel):
    scene_id: str
    check_id: str
    kendall_tau_b: float | None
    n_levels: int
    split: str


class HeldoutCheck(ArtifactModel):
    """Rates of the frozen tau on held-out clips; ``status`` is the fit's (not ok: indicative)."""

    check_id: str
    status: CheckStatus
    tau: float | None
    fpr_clean: Rate
    fpr_clean_decided: Rate
    fpr_off_target: Rate
    tpr_by_level: tuple[LevelTpr, ...]


class RealFpr(ArtifactModel):
    dataset: str
    check_id: str
    fpr: Rate


class EvaluationStratum(ArtifactModel):
    level: InputLevel
    profile: str
    checks: tuple[HeldoutCheck, ...]
    scene_kendall: tuple[SceneKendall, ...]
    real: tuple[RealFpr, ...]
    real_clip_flags: tuple[RealFpr, ...]
    """Share of real clips whose aggregate score is flagged (check_id ``score``)."""


class EvaluationReport(ArtifactModel):
    schema_: str = Field(default=EVALUATION_SCHEMA, alias=SCHEMA_KEY)
    version: str = CALIBRATION_VERSION
    created_utc: str
    run_id: str
    artifact_path: str
    artifact_sha256: str
    artifact_run_id: str
    dataset: DatasetRef
    real_datasets: tuple[str, ...]
    strata: tuple[EvaluationStratum, ...]
    real_sets: tuple[DatasetRef, ...] = ()
    """The real clip sets the evaluation scored (empty before 1.2.0): what rule 1.5 checks."""
