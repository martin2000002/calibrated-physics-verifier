"""Diagnosis report models and their deterministic JSON.

Callers pass rows in a deterministic order; JSON keys are sorted and non-finite floats become
null, so a fixed clock and run id give byte-identical files.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field

from physverify.core.models import ReportModel
from physverify.datasets._base import DatasetSource
from physverify.runtime.runs import RunContext
from physverify.schema.checks import AbstainReason, Verdict
from physverify.schema.clip import ClipManifest, Split
from physverify.schema.error_profile import ErrorProfile, no_profile_hash
from physverify.schema.stages import Stage
from physverify.stats.intervals import Interval

REPORT_DIR = "diagnosis"
REAL_VIDEO_SCOPE = (
    "Real videos have no per-stage ground truth: every number here is end-to-end error of the "
    "full perceived chain and is not attributable to a single stage."
)
MISSING = "n/a"
PROVISIONAL_NOTE = "provisional until the A8 threshold calibration"


def nominal_coverage(k: float) -> float:
    """Gaussian coverage of |z| <= k: 2 Phi(k) - 1."""
    return math.erf(k / math.sqrt(2.0))


def profile_hash(profile: ErrorProfile | None) -> str:
    if profile is None:
        return no_profile_hash()
    return profile.entries_hash(entry.key for entry in profile.entries)


class Coverage(ReportModel):
    k: float
    nominal: float
    observed: float
    n: int


def z_coverage(z: ArrayLike, levels: Iterable[float]) -> tuple[Coverage, ...]:
    """Observed share of finite ``|z| <= k`` next to its Gaussian nominal, per level ``k``."""
    values = np.abs(np.asarray(z, dtype=np.float64).ravel())
    values = values[np.isfinite(values)]
    n = int(values.size)
    return tuple(
        Coverage(
            k=float(k),
            nominal=nominal_coverage(float(k)),
            observed=float(np.mean(values <= k)) if n else math.nan,
            n=n,
        )
        for k in levels
    )


class Provenance(ReportModel):
    run_id: str
    commit: str
    config_hash: str
    profile: str
    device: str
    created_utc: str
    dataset: str
    dataset_versions: dict[str, str]
    subset: str
    error_profile_hash: str
    clips: tuple[str, ...]
    estimated: tuple[str, ...] = ()
    provisional: tuple[str, ...] = ()


def dataset_versions(dataset: DatasetSource, clips: Iterable[ClipManifest]) -> dict[str, str]:
    """``{dataset: version}`` from the ground-truth stores (several versions are comma-joined)."""
    versions: set[str] = set()
    for clip in clips:
        store = dataset.ground_truth(clip)
        if store is not None:
            versions.add(store.version)
    return {dataset.name: ",".join(sorted(versions))} if versions else {}


def build_provenance(
    run: RunContext,
    *,
    dataset: str,
    dataset_versions: Mapping[str, str],
    subset: str,
    error_profile: ErrorProfile | None,
    clips: Iterable[str],
    estimated: Iterable[str] = (),
    provisional: Iterable[str] = (),
) -> Provenance:
    return Provenance(
        run_id=run.run_id,
        commit=run.commit,
        config_hash=run.config_hash,
        profile=run.profile,
        device=run.device,
        created_utc=run.created_utc,
        dataset=dataset,
        dataset_versions=dict(dataset_versions),
        subset=subset,
        error_profile_hash=profile_hash(error_profile),
        clips=tuple(sorted(clips)),
        estimated=tuple(sorted(set(estimated))),
        provisional=tuple(sorted(set(provisional))),
    )


class Failure(ReportModel):
    clip_id: str
    error: str
    stage: Stage | None = None
    variant: str | None = None

    @classmethod
    def from_error(cls, clip_id: str, error: Exception, variant: str | None = None) -> Failure:
        return cls(clip_id=clip_id, error=f"{type(error).__name__}: {error}", variant=variant)


class StageMetricRow(ReportModel):
    """``values`` holds ``cases.summarize`` keys; ``coverage`` the z coverage per quantity."""

    stage: Stage
    metric: str
    provider: str
    pred_variant: str
    truth_variant: str
    split: Split
    n_clips: int
    values: dict[str, float]
    coverage: dict[str, tuple[Coverage, ...]] = Field(default_factory=dict)


class ProfileEntryRow(ReportModel):
    stage: Stage
    provider: str
    quantity: str
    bias: float
    sigma: float
    rms: float
    n: int
    written: bool


class StageErrorReport(ReportModel):
    kind: Literal["stage_errors"] = "stage_errors"
    provenance: Provenance
    profile_split: Split
    report_split: Split
    min_samples: int
    profile_written: bool
    profile_path: str | None = None
    rows: tuple[StageMetricRow, ...]
    profile: tuple[ProfileEntryRow, ...] = ()
    failures: tuple[Failure, ...] = ()


class FlipStats(ReportModel):
    """Verdict flips between two input levels; ``confusion[reference][other]`` counts."""

    available: bool
    reason: str | None = None
    provisional: bool = False
    rate: float | None = None
    n_decided: int = 0
    n_flips: int = 0
    confusion: dict[Verdict, dict[Verdict, int]] = Field(default_factory=dict)


class CheckAgreementRow(ReportModel):
    check_id: str
    unit: str
    n_pairs: int
    n_missing: int
    n_spurious: int
    bias: float
    mae: float
    rms: float
    rel_error: float
    n_z: int
    coverage: tuple[Coverage, ...]
    abstain_reference: dict[AbstainReason, float]
    abstain_other: dict[AbstainReason, float]
    flips: FlipStats


class Comparison(ReportModel):
    """``other`` measured against ``reference``, e.g. V0 against V8, or V8 against OracleSource."""

    name: str
    reference: str
    other: str
    checks: tuple[CheckAgreementRow, ...]


class ClipCheckRow(ReportModel):
    comparison: str
    clip_id: str
    check_id: str
    subject: tuple[str, ...]
    m_reference: float | None
    m_other: float | None
    delta: float | None
    verdict_reference: Verdict | None
    verdict_other: Verdict | None


class EndToEndReport(ReportModel):
    kind: Literal["end_to_end"] = "end_to_end"
    provenance: Provenance
    report_split: Split
    comparisons: tuple[Comparison, ...]
    clips: tuple[ClipCheckRow, ...] = ()
    failures: tuple[Failure, ...] = ()


class BudgetRow(ReportModel):
    row: str
    value: float
    percent: float
    ci: Interval | None = None
    stage_metric: str | None = None


class CheckAttribution(ReportModel):
    """Mean absolute error to the oracle per factorial cell, with Yates and Shapley splits."""

    check_id: str
    unit: str
    n_blocks: int
    n_clips: int
    cells: dict[str, float]
    cells_ci: dict[str, Interval] = Field(default_factory=dict)
    v0: float
    effects: dict[str, float]
    effects_ci: dict[str, Interval] = Field(default_factory=dict)
    shapley: dict[Stage, float]
    shapley_ci: dict[Stage, Interval] = Field(default_factory=dict)
    budget: tuple[BudgetRow, ...]
    incomplete: dict[str, int] = Field(default_factory=dict)
    bootstrap_dropped: int = 0


class AttributionReport(ReportModel):
    kind: Literal["attribution"] = "attribution"
    provenance: Provenance
    report_split: Split
    fixed_ground_truth: tuple[Stage, ...]
    lift_unit: str
    bootstrap: int
    seed: int
    ci: float
    checks: tuple[CheckAttribution, ...]
    failures: tuple[Failure, ...] = ()


class MotionRow(ReportModel):
    """Agreement with Physics-IQ pixel-change masks (not object accuracy)."""

    dataset: str
    clip_id: str
    scenario: str
    dynamic: bool
    n_frames: int
    motion_iou: float | None = None
    precision: float | None = None
    recall: float | None = None
    false_motion_rate: float | None = None
    unavailable: str | None = None


class FallRow(ReportModel):
    """Free-fall diagnosis of one clip; every g value is an estimate."""

    dataset: str
    clip_id: str
    object_id: str | None
    n_frames: int
    residual_ratio: float | None = None
    lateral_ratio: float | None = None
    g_est: float | None = None
    g_rel_error: float | None = None
    g_rel_sigma: float | None = None
    g_declared: float | None = None
    g_container: float | None = None
    closer_time_base: Literal["declared", "container"] | None = None
    gravity_dir_estimated: bool = False
    abstain_reason: AbstainReason | None = None


class VerdictDistribution(ReportModel):
    dataset: str
    check_id: str
    counts: dict[Verdict, int]
    fail_rate: float | None
    abstentions: dict[AbstainReason, int] = Field(default_factory=dict)


class RealVideoReport(ReportModel):
    kind: Literal["real"] = "real"
    error_scope: Literal["end_to_end_only"] = "end_to_end_only"
    provenance: Provenance
    motion: tuple[MotionRow, ...] = ()
    falls: tuple[FallRow, ...] = ()
    verdicts: tuple[VerdictDistribution, ...] = ()
    thresholds_provisional: bool = True
    failures: tuple[Failure, ...] = ()


Report = StageErrorReport | EndToEndReport | AttributionReport | RealVideoReport
