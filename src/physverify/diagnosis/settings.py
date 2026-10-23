"""The ``[diagnosis]`` configuration section (ADR 0007 §4.1, values in
configs/studies/diagnosis.toml)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from physverify.config.models import Settings
from physverify.core.models import FrozenModel
from physverify.diagnosis.chain_variants import FACTORS
from physverify.schema.clip import Split
from physverify.schema.stages import Stage

SECTION = "diagnosis"


class MetricSettings(FrozenModel):
    boundary_tol_frac: float = Field(gt=0)
    depth_alignment: Literal["clip", "frame"]
    depth_eps_frac: float = Field(gt=0)
    contact_min_tiou: float = Field(ge=0, le=1)
    impact_half_window_s: float = Field(ge=0)


class EndToEndSettings(FrozenModel):
    min_span_iou: float = Field(ge=0, le=1)
    z_levels: tuple[float, ...] = Field(min_length=1)


class FactorialSettings(FrozenModel):
    fixed_ground_truth: tuple[Stage, ...]
    bootstrap: int = Field(ge=0)
    seed: int
    ci: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _fixed_are_not_factors(self) -> FactorialSettings:
        clash = sorted(s.value for s in set(self.fixed_ground_truth) & set(FACTORS))
        if clash:
            raise ValueError(f"fixed_ground_truth overlaps the factorial factors: {clash}")
        return self


class ScaleDiagnosisSettings(FrozenModel):
    providers: tuple[str, ...] = Field(min_length=1)


class RealSettings(FrozenModel):
    datasets: tuple[str, ...]
    # Downloadable by name only: never part of the default real set, which they would seal.
    sealed_controls: tuple[str, ...] = ()
    motion_threshold: int = Field(ge=0, le=255)
    motion_min_frac: float = Field(ge=0, le=1)
    moving_iou: float = Field(ge=0, le=1)
    fall_speed_drop: float = Field(gt=0, le=1)
    min_fall_frames: int = Field(ge=3)


class DiagnosisSettings(FrozenModel):
    dataset: str
    subset: str
    profile_split: Split
    report_split: Split
    min_samples: int = Field(ge=1)
    metrics: MetricSettings
    end_to_end: EndToEndSettings
    factorial: FactorialSettings
    scale: ScaleDiagnosisSettings
    real: RealSettings


def diagnosis_settings(settings: Settings) -> DiagnosisSettings:
    return settings.section(SECTION, DiagnosisSettings)
