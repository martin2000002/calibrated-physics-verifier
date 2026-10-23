"""Grounding DINO tiny prompted once per query, boxes assigned to query slots."""

from __future__ import annotations

from typing import ClassVar

from physverify.extraction._base import WeightSpec
from physverify.extraction.providers._grounding_per_query import PromptedDetector
from physverify.extraction.providers.detect_grounding_dino_tiny import WEIGHTS


class GroundingDinoTinyPerQuery(PromptedDetector):
    name: ClassVar[str] = "grounding_dino_tiny_per_query"
    version: ClassVar[str] = "1"
    weights: ClassVar[tuple[WeightSpec, ...]] = (WEIGHTS,)
    spec: ClassVar[WeightSpec] = WEIGHTS


PLUGINS = (GroundingDinoTinyPerQuery,)
