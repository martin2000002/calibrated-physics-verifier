"""The ``[calibration]`` section (ADR 0009 §5) plus the values it reads from ``[verification]``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import Field, NonNegativeFloat

from physverify.checks._base import Check
from physverify.config.models import FrozenModel, Settings
from physverify.diagnosis.settings import diagnosis_settings
from physverify.schema.checks import InputLevel
from physverify.schema.io import stable_hash
from physverify.verification.policies.threshold import ThresholdParams
from physverify.verification.runner import (
    VERIFICATION_SECTION,
    ChecksSettings,
    VerificationSettings,
)

SECTION = "calibration"
THRESHOLD_POLICY = "threshold"


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


def checks_hash(registry: Mapping[str, type[Check]], overrides: ChecksSettings) -> str:
    """Hash of the effective parameters of every registered check."""
    params = {
        check_id: check.Params.model_validate(overrides.root.get(check_id, {})).model_dump(
            mode="json"
        )
        for check_id, check in sorted(registry.items())
    }
    return stable_hash(params)
