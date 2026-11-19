"""The files every command of a sealed look judges with, the abstention's k* among them."""

from __future__ import annotations

from pathlib import Path

from physverify.calibration.sealed_ledger import frozen_files
from physverify.config.models import Settings
from physverify.selective_risk.settings import ABSTENTION_FILE


def look_files(settings: Settings) -> dict[str, Path]:
    """The ledger's frozen files plus abstention.json: freeze.json records the same set, and a
    refit k* at the same commit is another freeze for every command of the look."""
    return {
        **frozen_files(settings),
        "abstention": settings.paths.calibration_dir / ABSTENTION_FILE,
    }
