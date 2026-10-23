"""Settings of the per-clip cost harness: the ``[timing]`` section of ``configs/timing.toml``.

Profiles (``[timing.profiles.<name>]``) are layered by the config loader, like every section.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.core.models import FrozenModel
from physverify.datasets.local_video import QueryList

SECTION = "timing"
SLUG_PATTERN = r"^[a-z0-9_]+$"
# WanPipeline: height and width divisible by 16, num_frames - 1 divisible by 4 (the VAE packs
# four frames into one latent frame after the first).
SPATIAL_MULTIPLE = 16
TEMPORAL_STRIDE = 4

WarmupScope = Literal["target", "clip"]
Dtype = Literal["bfloat16", "float16", "float32"]


class TargetRunSettings(FrozenModel):
    """Repetitions of one target. ``warmup_scope="clip"`` warms every clip, ``"target"`` once."""

    warmup: int = Field(0, ge=0)
    warmup_scope: WarmupScope = "clip"
    repetitions: int = Field(3, ge=1)
    max_clips: int | None = Field(None, ge=1)


class PromptSpec(FrozenModel):
    id: str = Field(pattern=SLUG_PATTERN)
    text: str = Field(min_length=1)
    queries: QueryList = Field(min_length=1)


class GenerationSettings(FrozenModel):
    """Generator call timed on the H200; must match what the fine-tuning cycles generate with."""

    model_ref: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    num_frames: int = Field(gt=0)
    fps: int = Field(gt=0)
    num_inference_steps: int = Field(ge=1)
    guidance_scale: float = Field(gt=0)
    flow_shift: float = Field(gt=0)
    batch_sizes: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)
    dtype: Dtype
    vae_dtype: Dtype
    seed: int = Field(0, ge=0)
    negative_prompt: str = ""
    subset: str = Field(pattern=SLUG_PATTERN)
    prompts: list[PromptSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> GenerationSettings:
        if self.height % SPATIAL_MULTIPLE or self.width % SPATIAL_MULTIPLE:
            raise ValueError(f"height and width must be multiples of {SPATIAL_MULTIPLE}")
        if (self.num_frames - 1) % TEMPORAL_STRIDE:
            raise ValueError(f"num_frames - 1 must be a multiple of {TEMPORAL_STRIDE}")
        if len(set(self.batch_sizes)) != len(self.batch_sizes) or 1 not in self.batch_sizes:
            raise ValueError("batch_sizes must be distinct and include 1")
        ids = [prompt.id for prompt in self.prompts]
        if len(set(ids)) != len(ids):
            raise ValueError("prompt ids must be unique")
        return self


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
