"""What ``baselines compare`` writes: the cut u_B1 (``cuts.json``) and, per part, every clip's label
with its B1 and judge score under the look's freeze (``kubric.json``, ``real.json``)."""

from __future__ import annotations

from typing import Final

from physverify.baselines.settings import Part
from physverify.core.models import ReportModel

CUTS_FILE: Final = "cuts.json"
PART_FILES: Final[dict[Part, str]] = {"kubric": "kubric.json", "real": "real.json"}


class ClipRow(ReportModel):
    """``scene`` is the resampling unit (the clip itself on a real set); ``judge`` is None when
    the judge gave no readable score."""

    clip_id: str
    scene: str
    perturbed: bool
    b1: float | None
    judge: float | None


class Cuts(ReportModel):
    """u_B1 from the calibration set's clean clips only (in-sample, development data)."""

    u_b1: float | None
    clean_fpr: float
    n_clean: int
    calibration_dataset: str
    calibration_sha256: str
    excluded_checks: tuple[str, ...]


class SetRows(ReportModel):
    dataset: str
    subset: str | None
    scores_sha256: str
    stamps: tuple[str, ...]
    rows: tuple[ClipRow, ...]


class PartRows(ReportModel):
    """``freeze`` and ``looks`` (ledger lines of this command under that freeze, this one
    included) are None and 0 on development sets, which write no ledger line."""

    part: Part
    judge: str
    cut: float
    cuts: Cuts
    cuts_sha256: str
    freeze: str | None
    looks: int
    sets: tuple[SetRows, ...]
