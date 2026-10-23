"""Settings of the Wan2.1 generator: the ``[timing.generation]`` table of the timing config."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from physverify.core.models import FrozenModel
from physverify.datasets.local_video import QueryList

SLUG_PATTERN = r"^[a-z0-9_]+$"
# WanPipeline: height and width divisible by 16, num_frames - 1 divisible by 4 (the VAE packs
# four frames into one latent frame after the first).
SPATIAL_MULTIPLE = 16
TEMPORAL_STRIDE = 4

Dtype = Literal["bfloat16", "float16", "float32"]


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
