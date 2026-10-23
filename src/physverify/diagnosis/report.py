"""Diagnosis report models, deterministic JSON and markdown tables (ADR 0007 §4.1).

Callers pass rows in a deterministic order; JSON keys are sorted and non-finite floats become
null, so a fixed clock and run id give byte-identical files.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from functools import singledispatch
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from pydantic import Field

from physverify.core.markdown import number, table
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


def write_report(run_dir: Path, name: str, report: Report) -> tuple[Path, Path]:
    """Write ``run_dir/diagnosis/<name>.json`` and ``.md``; returns both paths."""
    directory = run_dir / REPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    json_path, md_path = directory / f"{name}.json", directory / f"{name}.md"
    json_path.write_text(report.to_json(), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def render_markdown(report: Report) -> str:
    return "\n".join(_body(report)).rstrip("\n") + "\n"


def _cell(value: object) -> str:
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return number(value, ".4g", MISSING, "nonfinite")
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, tuple):
        return ", ".join(_cell(item) for item in value) or "-"
    return str(value).replace("|", "\\|")


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    return [*table(headers, ([_cell(value) for value in cells] for cells in rows)), ""]


def _section(title: str, lines: Sequence[str]) -> list[str]:
    return [f"## {title}", "", *lines] if lines else []


def _provenance(p: Provenance) -> list[str]:
    versions = ", ".join(f"{k} {v}" for k, v in sorted(p.dataset_versions.items())) or MISSING
    lines = [
        f"- run `{p.run_id}`, commit `{p.commit or MISSING}`, profile `{p.profile}`, "
        f"device `{p.device}`, created {p.created_utc}",
        f"- config hash `{p.config_hash}`, error profile hash `{p.error_profile_hash}`",
        f"- dataset `{p.dataset}` ({versions}), subset `{p.subset}`, {len(p.clips)} clips",
    ]
    if p.estimated:
        lines.append(f"- [estimated] (not exact measurements): {', '.join(p.estimated)}")
    if p.provisional:
        lines.append(f"- [provisional] {PROVISIONAL_NOTE}: {', '.join(p.provisional)}")
    return [*lines, ""]


def _failures(failures: Sequence[Failure]) -> list[str]:
    rows = [(f.clip_id, f.variant, f.stage, f.error) for f in failures]
    return _section("Failures", _table(("clip", "variant", "stage", "error"), rows) if rows else [])


def _header(title: str, provenance: Provenance) -> list[str]:
    return [f"# {title}", "", *_provenance(provenance)]


@singledispatch
def _body(report: ReportModel) -> list[str]:
    raise TypeError(f"no markdown renderer for {type(report).__name__}")


SAMPLE_STATS = ("mean", "std", "rms", "n")


def _split_values(values: Mapping[str, float]) -> tuple[list[list[object]], list[list[object]]]:
    samples: dict[str, dict[str, float]] = {}
    ratios: dict[str, dict[str, float]] = {}
    for key, value in values.items():
        base, _, stat = key.rpartition(".")
        if stat in SAMPLE_STATS:
            samples.setdefault(base, {})[stat] = value
        elif stat in ("num", "den"):
            ratios.setdefault(base, {})[stat] = value
        else:
            ratios.setdefault(key, {})["value"] = value
    sample_rows: list[list[object]] = [
        [q, *(s.get(stat) for stat in SAMPLE_STATS)] for q, s in sorted(samples.items())
    ]
    ratio_rows: list[list[object]] = [
        [r, s.get("value"), s.get("num"), s.get("den")] for r, s in sorted(ratios.items())
    ]
    return sample_rows, ratio_rows


def _metric_row(row: StageMetricRow) -> list[str]:
    samples, ratios = _split_values(row.values)
    lines = [
        f"### {row.metric} ({row.split.value}): `{row.provider}`, "
        f"{row.pred_variant} vs {row.truth_variant}, {row.n_clips} clips",
        "",
    ]
    if samples:
        lines += _table(("quantity", "mean", "std", "rms", "n"), samples)
    if ratios:
        lines += _table(("rate", "value", "num", "den"), ratios)
    coverage = [(q, _coverage_text(c)) for q, c in sorted(row.coverage.items())]
    if coverage:
        lines += _table(("quantity", "z coverage"), coverage)
    return lines


@_body.register
def _stage_body(report: StageErrorReport) -> list[str]:
    status = "written" if report.profile_written else "kept in memory only"
    lines = _header("Per-stage extractor errors (Level 1)", report.provenance)
    lines += [
        f"Profile from split `{report.profile_split.value}` ({status}"
        + (f": `{report.profile_path}`" if report.profile_path else "")
        + f"), report split `{report.report_split.value}`, "
        f"entries need n >= {report.min_samples}.",
        "",
    ]
    for row in report.rows:
        lines += _metric_row(row)
    profile_rows = [
        (e.stage, e.provider, e.quantity, e.bias, e.sigma, e.rms, e.n, e.written)
        for e in report.profile
    ]
    headers = ("stage", "provider", "quantity", "bias", "sigma", "rms", "n", "written")
    entries = _table(headers, profile_rows) if profile_rows else []
    lines += _section("Error profile entries", entries)
    return lines + _failures(report.failures)


def _flip_text(flips: FlipStats) -> str:
    if not flips.available:
        return f"unavailable: {flips.reason or 'no verdict policy'}"
    text = f"{_cell(flips.rate)} ({flips.n_flips}/{flips.n_decided})"
    return f"{text} [provisional]" if flips.provisional else text


def _coverage_text(coverage: Sequence[Coverage]) -> str:
    return "; ".join(
        f"|z|<={_cell(c.k)}: {_cell(c.observed)} (nominal {_cell(c.nominal)})" for c in coverage
    )


def _rates_text(rates: Mapping[AbstainReason, float]) -> str:
    return ", ".join(f"{r.value} {_cell(v)}" for r, v in sorted(rates.items())) or "-"


def _comparison(comparison: Comparison) -> list[str]:
    headers = ("check", "unit", "pairs", "missing", "spurious", "bias", "MAE", "RMS", "rel err")
    rows = [_agreement(c) for c in comparison.checks]
    detail = [
        (
            c.check_id,
            c.n_z,
            _coverage_text(c.coverage),
            _rates_text(c.abstain_reference),
            _rates_text(c.abstain_other),
        )
        for c in comparison.checks
    ]
    title = f"{comparison.name}: {comparison.other} against {comparison.reference}"
    lines = _table(headers, rows)
    lines += _table(("check", "n z", "coverage", "abstain ref", "abstain other"), detail)
    return _section(title, lines)


def _agreement(c: CheckAgreementRow) -> tuple[object, ...]:
    return (
        c.check_id,
        c.unit,
        c.n_pairs,
        c.n_missing,
        c.n_spurious,
        c.bias,
        c.mae,
        c.rms,
        c.rel_error,
    )


def _flip_table(comparisons: Sequence[Comparison]) -> list[str]:
    check_ids = sorted({c.check_id for comparison in comparisons for c in comparison.checks})
    by_name = {cmp.name: {c.check_id: c for c in cmp.checks} for cmp in comparisons}
    rows = [
        [check_id, *(_row_flip(by_name[cmp.name].get(check_id)) for cmp in comparisons)]
        for check_id in check_ids
    ]
    headers = ("check", *(cmp.name for cmp in comparisons))
    return _section("Verdict flip rates (side by side)", _table(headers, rows) if rows else [])


def _row_flip(row: CheckAgreementRow | None) -> str:
    return MISSING if row is None else _flip_text(row.flips)


@_body.register
def _end_to_end_body(report: EndToEndReport) -> list[str]:
    lines = _header("End-to-end check error (Level 2)", report.provenance)
    lines += [f"Report split `{report.report_split.value}`.", ""]
    for comparison in report.comparisons:
        lines += _comparison(comparison)
    lines += _flip_table(report.comparisons)
    clip_rows = [
        (
            r.comparison,
            r.clip_id,
            r.check_id,
            r.subject,
            r.m_reference,
            r.m_other,
            r.delta,
            r.verdict_reference,
            r.verdict_other,
        )
        for r in report.clips
    ]
    headers = (
        "comparison",
        "clip",
        "check",
        "subject",
        "m ref",
        "m other",
        "delta",
        "v ref",
        "v other",
    )
    lines += _section("Per clip", _table(headers, clip_rows) if clip_rows else [])
    return lines + _failures(report.failures)


def _named(values: Mapping[Any, float], intervals: Mapping[Any, Interval]) -> list[list[object]]:
    rows: list[list[object]] = []
    for key in sorted(values, key=_cell):
        interval = intervals.get(key)
        bounds = (interval.low, interval.high) if interval else (None, None)
        rows.append([key, values[key], *bounds])
    return rows


def _attribution(check: CheckAttribution, ci: float) -> list[str]:
    bounds = (f"{_cell(100 * ci)}% low", "high")
    budget = [
        (
            b.row,
            b.value,
            b.percent,
            *((b.ci.low, b.ci.high) if b.ci else (None, None)),
            b.stage_metric,
        )
        for b in check.budget
    ]
    incomplete = ", ".join(f"{k} {v}" for k, v in sorted(check.incomplete.items())) or "none"
    lines = [
        f"{check.n_blocks} complete blocks over {check.n_clips} clips, unit `{check.unit}`, "
        f"V0 mean error {_cell(check.v0)}. Incomplete blocks: {incomplete}. "
        f"Dropped bootstrap replicas: {check.bootstrap_dropped}.",
        "",
    ]
    lines += _table(("cell", "mean abs error", *bounds), _named(check.cells, check.cells_ci))
    lines += _table(("effect", "value", *bounds), _named(check.effects, check.effects_ci))
    lines += _table(("stage", "Shapley", *bounds), _named(check.shapley, check.shapley_ci))
    lines += _table(("budget row", "value", "%", *bounds, "Level 1 metric"), budget)
    return _section(f"{check.check_id}", lines)


@_body.register
def _attribution_body(report: AttributionReport) -> list[str]:
    lines = _header("Error attribution by stage (Level 3)", report.provenance)
    fixed = ", ".join(stage.value for stage in report.fixed_ground_truth)
    lines += [
        f"Report split `{report.report_split.value}`. Fixed ground truth: {fixed}. "
        f"Clip bootstrap: {report.bootstrap} replicas, seed {report.seed}, "
        f"{_cell(100 * report.ci)}% intervals. LIFT unit in every cell: `{report.lift_unit}`.",
        "",
    ]
    for check in report.checks:
        lines += _attribution(check, report.ci)
    return lines + _failures(report.failures)


def _motion(rows: Sequence[MotionRow]) -> list[str]:
    headers = (
        "dataset",
        "clip",
        "scenario",
        "dynamic",
        "frames",
        "motion IoU",
        "precision",
        "recall",
        "false motion",
        "unavailable",
    )
    body = [
        (
            r.dataset,
            r.clip_id,
            r.scenario,
            r.dynamic,
            r.n_frames,
            r.motion_iou,
            r.precision,
            r.recall,
            r.false_motion_rate,
            r.unavailable,
        )
        for r in rows
    ]
    note = "Physics-IQ masks mark pixel change, not objects: IoU is agreement, not accuracy."
    return _section("Motion agreement", [note, "", *_table(headers, body)] if body else [])


def _falls(rows: Sequence[FallRow]) -> list[str]:
    headers = (
        "dataset",
        "clip",
        "object",
        "frames",
        "residual ratio",
        "lateral ratio",
        "g [estimated]",
        "g rel error [estimated]",
        "g rel sigma",
        "g declared dt",
        "g container dt",
        "closer",
        "gravity dir",
        "abstain",
    )
    body = [
        (
            r.dataset,
            r.clip_id,
            r.object_id,
            r.n_frames,
            r.residual_ratio,
            r.lateral_ratio,
            r.g_est,
            r.g_rel_error,
            r.g_rel_sigma,
            r.g_declared,
            r.g_container,
            r.closer_time_base,
            "[estimated] camera +y" if r.gravity_dir_estimated else "states",
            r.abstain_reason,
        )
        for r in rows
    ]
    return _section("Free fall", _table(headers, body) if body else [])


def _verdicts(rows: Sequence[VerdictDistribution], provisional: bool) -> list[str]:
    verdicts = tuple(Verdict)
    headers = ("dataset", "check", *(v.value for v in verdicts), "fail rate", "abstentions")
    body = [
        (
            r.dataset,
            r.check_id,
            *(r.counts.get(v, 0) for v in verdicts),
            r.fail_rate,
            _counts_text(r.abstentions),
        )
        for r in rows
    ]
    note = f"Thresholds are {PROVISIONAL_NOTE}." if provisional else "Thresholds are calibrated."
    return _section("Verdicts on real video", [note, "", *_table(headers, body)] if body else [])


def _counts_text(counts: Mapping[AbstainReason, int]) -> str:
    return ", ".join(f"{r.value} {n}" for r, n in sorted(counts.items())) or "-"


@_body.register
def _real_body(report: RealVideoReport) -> list[str]:
    lines = ["# Real video diagnosis (end-to-end only)", "", REAL_VIDEO_SCOPE, ""]
    lines += _provenance(report.provenance)
    lines += _motion(report.motion)
    lines += _falls(report.falls)
    lines += _verdicts(report.verdicts, report.thresholds_provisional)
    return lines + _failures(report.failures)
