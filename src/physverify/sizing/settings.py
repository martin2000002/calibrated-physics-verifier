"""The ``[sizing]`` configuration section: Objectron fetch, fit, lookup tables."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from physverify.config.models import Settings
from physverify.core.models import FrozenModel

SECTION = "sizing"


class FixedSize(FrozenModel):
    """A curated regulation size (an estimate: the spread covers the allowed range)."""

    mean_m: float = Field(gt=0)
    rel_spread: float = Field(ge=0.05)
    source: str = Field(min_length=1)


class SizeLookup(FrozenModel):
    """The part of ``[sizing]`` that changes a predicted size: the provider's cache key covers
    these fields only (``extra = ignore`` drops the fetch and fit keys)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    labels: dict[str, str] = Field(default_factory=dict)
    fixed: dict[str, FixedSize] = Field(default_factory=dict)


class EndToEndSettings(FrozenModel):
    classes: tuple[str, ...] = ()
    videos: int = Field(default=20, ge=1)
    seconds: float = Field(default=2.0, gt=0)
    fps: float = Field(default=15.0, gt=0)
    max_side: int = Field(default=640, ge=64)
    max_video_mb: float = Field(default=40.0, gt=0)
    subset: str = "objectron"
    min_box_iou: float = Field(default=0.3, ge=0, le=1)


class SizingSettings(SizeLookup):
    model_config = ConfigDict(extra="forbid")

    bucket_url: str = "https://storage.googleapis.com/objectron"
    listing_url: str = "https://storage.googleapis.com/storage/v1/b/objectron/o"
    license: str = "C-UDA-1.0"
    classes: tuple[str, ...] = ()
    per_class_train: int = Field(default=150, ge=1)
    per_class_test: int = Field(default=60, ge=1)
    seed: int = 0
    folds: int = Field(default=5, ge=2)
    min_extent_m: float = Field(default=0.01, ge=0)
    coverage_target: float = Field(default=0.683, gt=0, lt=1)
    baseline_prior: str = "cube"
    head_bytes: int = Field(default=8192, ge=64)
    workers: int = Field(default=8, ge=1)
    retries: int = Field(default=3, ge=0)
    bootstrap: int = Field(default=1000, ge=1)
    ci: float = Field(default=0.95, gt=0, lt=1)
    e2e: EndToEndSettings = Field(default_factory=EndToEndSettings)


def sizing_settings(settings: Settings) -> SizingSettings:
    return settings.section(SECTION, SizingSettings)
