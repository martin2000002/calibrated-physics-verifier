"""The ``[extraction]`` configuration section (ADR 0005, 3.1)."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from physverify.core.models import FrozenModel
from physverify.schema.stages import Stage

SECTION = "extraction"


class SamplingSettings(FrozenModel):
    max_side: int | None = Field(default=640, gt=0)
    target_fps: dict[str, float] = Field(default_factory=dict)
    start: int = Field(default=0, ge=0)
    stop: int | None = Field(default=None, gt=0)


class ExtractionSettings(FrozenModel):
    """Stage -> provider plan, sampling and per-provider parameter overrides."""

    plan: dict[Stage, str]
    sampling: SamplingSettings = SamplingSettings()
    providers: dict[Stage, dict[str, dict[str, Any]]] = Field(default_factory=dict)
