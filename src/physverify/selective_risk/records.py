"""The JSON records of the selective-risk study: ``abstention.json`` and
``abstention_report.json``. Field names are the persisted keys."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer

from physverify.core.models import FrozenModel
from physverify.verification.thresholding import ArtifactRef


class SparseModel(FrozenModel):
    """Leaves its None fields out of the record (the key is absent, not null)."""

    @model_serializer(mode="wrap")
    def _omit_none(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        dumped: dict[str, object] = handler(self)
        return {key: value for key, value in dumped.items() if value is not None}


class Counts(FrozenModel):
    cases: int
    clips: int
    clips_failed: int
    events_without_reference: int
    spurious_perceived: int
    missed_by_perception: int


class TableRef(FrozenModel):
    """A threshold table by its label, artifact and the hash of its content."""

    status: str
    source: str
    artifact: ArtifactRef | None
    table_hash: str


class PointRow(FrozenModel):
    k_sigma: float
    alpha: float
    n: int
    decided: int
    coverage: float
    risk: float
    risk_ci: tuple[float, float]
    false_fail_rate: float
    missed_fail_rate: float


class SweptTable(FrozenModel):
    """One table swept over the k grid; ``best`` indexes ``curve``."""

    curve: tuple[PointRow, ...]
    best: int | None
    alpha_star: float | None
    k_star: float | None


class AurcDifference(FrozenModel):
    value: float
    ci: tuple[float | None, float | None]
    n_boot_non_finite: int | None


class ComparedTables(FrozenModel):
    raw: SweptTable
    calibrated: SweptTable
    aurc_calibrated_minus_raw: AurcDifference
    identical_tables: bool


class InjectedLabel(SparseModel):
    rows: int
    at: PointRow | None = None
    oracle_label_disagreements: tuple[str, ...] | None = None


class ZSummary(SparseModel):
    n: int
    quantiles: dict[str, float] | None = None


class RealRows(SparseModel):
    n: Counts
    k_star: PointRow | None = None
    k_2: PointRow


class RealError(FrozenModel):
    error: str


class AbstentionRecord(FrozenModel):
    """``abstention.json``: k* fitted on the calib cases."""

    estimated: Literal[True] = True
    dataset: str
    split: str
    risk_target: float
    ci: float
    alpha_star: float | None
    k_star: float | None
    n: Counts
    thresholds: dict[str, TableRef]
    curve: tuple[PointRow, ...]


class FittedK(BaseModel):
    """What ``selective-risk curve`` reads back from ``abstention.json``."""

    model_config = ConfigDict(frozen=True, extra="ignore")
    k_star: float | None = None


class CurveRecord(FrozenModel):
    """``abstention_report.json``: both tables on the eval cases."""

    estimated: Literal[True] = True
    split: str
    k_star_from_calib: float | None
    n: Counts
    thresholds: dict[str, TableRef]
    tables: ComparedTables
    per_check: dict[str, dict[str, PointRow]]
    injected_label: InjectedLabel
    z: ZSummary
    real_video: dict[str, RealRows | RealError]
