"""The ``[abstention]`` settings and the files the selective-risk study writes."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from physverify.calibration.settings import calibration_settings
from physverify.config.models import Settings
from physverify.core.models import FrozenModel

SECTION = "abstention"
ABSTENTION_FILE = "abstention.json"
"""k* fitted on calib (``selective-risk fit``), under ``paths.calibration_dir``."""
REPORT_FILE = "abstention_report.json"
CURVE_FILE = "curve.csv"
"""``selective-risk curve`` writes both into its run directory."""


class AbstentionSettings(FrozenModel):
    risk_target: float = Field(default=0.10, gt=0, lt=1)
    ci: float = Field(default=0.95, gt=0, lt=1)
    alpha_max: float = Field(default=0.5, gt=0, le=0.5)
    alpha_min: float = Field(default=1e-4, gt=0)
    grid_points: int = Field(default=25, ge=2)
    bootstrap: int = Field(default=1000, ge=1)
    seed: int = 0


@dataclass(frozen=True)
class Population:
    """The clips the study reads: those of the calibration chain (``[calibration]``), its calib
    split for ``fit`` and its eval split for ``curve``, and its real controls."""

    dataset: str
    subset: str
    real_datasets: tuple[str, ...]


def population(settings: Settings) -> Population:
    own = calibration_settings(settings)
    return Population(own.dataset, own.subset, own.real_datasets)
