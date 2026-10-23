"""The ``[calibration]`` section plus the values it reads from ``[verification]``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, NonNegativeFloat

from physverify.config.models import Settings
from physverify.core.models import FrozenModel
from physverify.diagnosis.settings import diagnosis_settings
from physverify.schema.checks import InputLevel
from physverify.verification.policies.threshold import ThresholdParams
from physverify.verification.runner import VERIFICATION_SECTION, VerificationSettings

SECTION = "calibration"
THRESHOLD_POLICY = "threshold"
PairFilter = Literal["all", "nonzero"]


class CalibrationSettings(FrozenModel):
    dataset: str = "kubric_synthetic"
    subset: str = "full"
    levels: tuple[InputLevel, ...] = (InputLevel.ORACLE, InputLevel.PERCEIVED)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    beta: float = Field(default=0.05, gt=0, lt=1)
    folds: int = Field(default=5, ge=2)
    bootstrap: int = Field(default=2000, ge=0)
    seed: int = 0
    ci: float = Field(default=0.95, gt=0, lt=1)
    min_sigma_pairs: int = Field(default=10, ge=2)
    sigma_pair_filter: PairFilter
    """Per-check sigma factor k_c over every usable pair ("all", the estimator of the fits made
    before October 20, 2026) or only over the pairs with a nonzero magnitude on some side
    ("nonzero"); the bias b_c always uses every usable pair. Required: a config without it must not
    fall back silently to either estimator."""
    platt_l2: float = Field(default=0.01, ge=0)
    ld_max_extrapolation: float = Field(default=1.0, ge=0)
    real_datasets: tuple[str, ...] = ("physics_iq", "pisa_real")
    artifact: str = "calibration.json"
    stage_profile: str = "stage_errors.recalibrated.json"
    tau_tolerance: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    """epsilon_c per check in check units: tau = max(q, epsilon_c); absent = 0."""
    require_calibrated_artifact: bool = False
    """A consumer also requires artifact status ``calibrated``, not only a usable stratum."""


@dataclass(frozen=True)
class CalibrationConfig:
    """Everything a fit reads: its own section, the verdict rule, couplings and pairing IoU."""

    calibration: CalibrationSettings
    policy: ThresholdParams
    couplings: frozenset[tuple[str, str]]
    min_span_iou: float


def calibration_settings(settings: Settings) -> CalibrationSettings:
    return settings.section(SECTION, CalibrationSettings)


def calibration_config(settings: Settings) -> CalibrationConfig:
    verification = settings.section(VERIFICATION_SECTION, VerificationSettings)
    raw = verification.policies.get(THRESHOLD_POLICY, {})
    return CalibrationConfig(
        calibration=calibration_settings(settings),
        policy=ThresholdParams.model_validate(raw),
        couplings=frozenset((p, c) for p, c in verification.specificity.couplings),
        min_span_iou=diagnosis_settings(settings).end_to_end.min_span_iou,
    )
