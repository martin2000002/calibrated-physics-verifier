"""``[agreement]`` (configs/studies/human_agreement.toml): the pre-registered comparison of the
human-agreement study."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from physverify.core.models import FrozenModel
from physverify.datasets._base import DEFAULT_LABEL_SOURCE
from physverify.human_agreement.clip_decision import RULES
from physverify.human_agreement.undecided_policy import POLICIES

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
    permutation_exact_max: int = Field(ge=0)
    """Discordant captions up to which the secondary caption McNemar is exact (2^m flips)."""
    error_sample_per_class: int = Field(ge=0)
    error_seed: int
    on_unusable_stratum: Literal["descriptive", "error"] = "descriptive"
    """A calibrated rule whose stratum cannot score yet: fall back to the provisional rule and
    report the study as not evaluable, or stop with a configuration error (item 1.3)."""
    exclude_time_dependent: bool = False
    """Leave [check_scope].time_dependent_checks out of the primary rule; the primary keeps them
    (true is a pre-registration change, declared before the freeze)."""
    dev_subset: str
    test_subset: str
    preregistration: str
    label_source: str = DEFAULT_LABEL_SOURCE
    """The dataset plugin whose human labels the study compares with (a ``HumanLabelSource``)."""

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
