"""The ``[extraction]`` configuration section (ADR 0005, 3.1)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from physverify.schema.stages import Stage

SECTION = "extraction"


class SamplingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_side: int | None = Field(default=640, gt=0)
    target_fps: dict[str, float] = Field(default_factory=dict)
    start: int = Field(default=0, ge=0)
    stop: int | None = Field(default=None, gt=0)


class ExtractionSettings(BaseModel):
    """Stage -> provider plan, sampling and per-provider parameter overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: dict[Stage, str]
    sampling: SamplingSettings = SamplingSettings()
    providers: dict[Stage, dict[str, dict[str, Any]]] = Field(default_factory=dict)
