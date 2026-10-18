"""``[agreement]`` (configs/agreement.toml): the pre-registered comparison of ADR 0011."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from physverify.agreement.abstention import POLICIES
from physverify.agreement.decision import RULES
from physverify.config.models import FrozenModel

if TYPE_CHECKING:
    from pathlib import Path

    from physverify.config.models import Settings

SECTION = "agreement"
THRESHOLD_POLICY = "threshold"


class AgreementSettings(FrozenModel):
    rule: str
    primary_policy: str
    alpha: float = Field(gt=0, lt=1)
    power: float = Field(gt=0, lt=1)
    bootstrap_n: int = Field(ge=0)
    bootstrap_seed: int
    ci: float = Field(gt=0, lt=1)
    error_sample_per_class: int = Field(ge=0)
    error_seed: int
    exclude_checks: tuple[str, ...] = ()
    time_dependent_checks: tuple[str, ...] = ()
    dev_subset: str
    test_subset: str
    preregistration: str

    @model_validator(mode="after")
    def _known(self) -> AgreementSettings:
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        if self.primary_policy not in POLICIES:
            raise ValueError(f"primary_policy must be one of {POLICIES}")
        return self


def agreement_settings(settings: Settings) -> AgreementSettings:
    return settings.section(SECTION, AgreementSettings)


def preregistration_path(settings: Settings, agreement: AgreementSettings) -> Path:
    return settings.paths.data_dir / agreement.preregistration
