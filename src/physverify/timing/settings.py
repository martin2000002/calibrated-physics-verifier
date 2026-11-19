"""Settings of the per-clip cost harness: the ``[timing]`` section of
``configs/studies/timing.toml``.

Profiles (``[timing.profiles.<name>]``) are layered by the config loader, like every section.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, ValidationError, model_validator

from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.core.models import FrozenModel
from physverify.generation.settings import GenerationSettings

SECTION = "timing"

WarmupScope = Literal["target", "clip"]


class TargetRunSettings(FrozenModel):
    """Repetitions of one target. ``warmup_scope="clip"`` warms every clip, ``"target"`` once."""

    warmup: int = Field(0, ge=0)
    warmup_scope: WarmupScope = "clip"
    repetitions: int = Field(3, ge=1)
    max_clips: int | None = Field(None, ge=1)


class TimingSettings(FrozenModel):
    order: list[str] = Field(min_length=1)
    targets: dict[str, TargetRunSettings] = Field(default_factory=dict)
    valid_profile: str = "precise"
    max_iqr_rel: float = Field(0.10, gt=0)
    total_measures: list[str] = Field(min_length=1)
    generation: GenerationSettings

    @model_validator(mode="after")
    def _check(self) -> TimingSettings:
        for label, names in (("order", self.order), ("total_measures", self.total_measures)):
            if len(set(names)) != len(names):
                raise ValueError(f"{label} has repeated entries")
        return self


def load_timing(settings: Settings) -> TimingSettings:
    return settings.section(SECTION, TimingSettings)


def run_settings(timing: TimingSettings, name: str) -> TargetRunSettings:
    return timing.targets.get(name, TargetRunSettings())


def with_overrides(
    run: TargetRunSettings, *, warmup: int | None, repetitions: int | None
) -> TargetRunSettings:
    """Command-line overrides, validated like the configuration."""
    updates = {
        key: value
        for key, value in (("warmup", warmup), ("repetitions", repetitions))
        if value is not None
    }
    try:
        return TargetRunSettings.model_validate({**run.model_dump(), **updates})
    except ValidationError as exc:
        raise UsageError(f"invalid --warmup/--repetitions: {exc}") from exc
