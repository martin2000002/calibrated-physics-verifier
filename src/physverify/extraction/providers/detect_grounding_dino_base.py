"""Grounding DINO base, joint or per-query prompt."""

from __future__ import annotations

from typing import ClassVar

from physverify.extraction._base import WeightKind, WeightSpec
from physverify.extraction.providers._grounding_per_query import PromptedDetector

BASE_WEIGHTS = WeightSpec(
    WeightKind.HF,
    "IDEA-Research/grounding-dino-base",
    "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
    "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21",
)


class GroundingDinoBase(PromptedDetector):
    name: ClassVar[str] = "grounding_dino_base"
    version: ClassVar[str] = "1"
    weights: ClassVar[tuple[WeightSpec, ...]] = (BASE_WEIGHTS,)
    spec: ClassVar[WeightSpec] = BASE_WEIGHTS


PLUGINS = (GroundingDinoBase,)
