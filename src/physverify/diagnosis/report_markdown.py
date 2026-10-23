"""Markdown rendering of the diagnosis reports and the writer of their JSON and markdown files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import singledispatch
from pathlib import Path
from typing import Any

from physverify.core.models import ReportModel
from physverify.diagnosis.markdown_cells import cell_text, section, text_table
from physverify.diagnosis.real_video_markdown import fall_section, motion_section, verdict_section
from physverify.diagnosis.report import (
    MISSING,
    PROVISIONAL_NOTE,
    REAL_VIDEO_SCOPE,
    REPORT_DIR,
    AttributionReport,
    CheckAgreementRow,
    CheckAttribution,
    Comparison,
    Coverage,
    EndToEndReport,
    Failure,
    FlipStats,
    Provenance,
    RealVideoReport,
    Report,
    StageErrorReport,
    StageMetricRow,
)
from physverify.schema.checks import AbstainReason
from physverify.stats.intervals import Interval


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


def _provenance(p: Provenance) -> list[str]:
    versions = ", ".join(f"{k} {v}" for k, v in sorted(p.dataset_versions.items())) or MISSING
    lines = [
        (
            f"- run `{p.run_id}`, commit `{p.commit or MISSING}`, profile `{p.profile}`, "
            f"device `{p.device}`, created {p.created_utc}"
        ),
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
    return section(
        "Failures", text_table(("clip", "variant", "stage", "error"), rows) if rows else []
    )


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
        (
            f"### {row.metric} ({row.split.value}): `{row.provider}`, "
            f"{row.pred_variant} vs {row.truth_variant}, {row.n_clips} clips"
        ),
        "",
    ]
    if samples:
        lines += text_table(("quantity", "mean", "std", "rms", "n"), samples)
    if ratios:
        lines += text_table(("rate", "value", "num", "den"), ratios)
    coverage = [(q, _coverage_text(c)) for q, c in sorted(row.coverage.items())]
    if coverage:
        lines += text_table(("quantity", "z coverage"), coverage)
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
    entries = text_table(headers, profile_rows) if profile_rows else []
    lines += section("Error profile entries", entries)
    return lines + _failures(report.failures)


def _flip_text(flips: FlipStats) -> str:
    if not flips.available:
        return f"unavailable: {flips.reason or 'no verdict policy'}"
    text = f"{cell_text(flips.rate)} ({flips.n_flips}/{flips.n_decided})"
    return f"{text} [provisional]" if flips.provisional else text


def _coverage_text(coverage: Sequence[Coverage]) -> str:
    return "; ".join(
        f"|z|<={cell_text(c.k)}: {cell_text(c.observed)} (nominal {cell_text(c.nominal)})"
        for c in coverage
    )


def _rates_text(rates: Mapping[AbstainReason, float]) -> str:
    return ", ".join(f"{r.value} {cell_text(v)}" for r, v in sorted(rates.items())) or "-"


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
    lines = text_table(headers, rows)
    lines += text_table(("check", "n z", "coverage", "abstain ref", "abstain other"), detail)
    return section(title, lines)


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
    return section("Verdict flip rates (side by side)", text_table(headers, rows) if rows else [])


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
    lines += section("Per clip", text_table(headers, clip_rows) if clip_rows else [])
    return lines + _failures(report.failures)


def _named(values: Mapping[Any, float], intervals: Mapping[Any, Interval]) -> list[list[object]]:
    rows: list[list[object]] = []
    for key in sorted(values, key=cell_text):
        interval = intervals.get(key)
        bounds = (interval.low, interval.high) if interval else (None, None)
        rows.append([key, values[key], *bounds])
    return rows


def _attribution(check: CheckAttribution, ci: float) -> list[str]:
    bounds = (f"{cell_text(100 * ci)}% low", "high")
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
        (
            f"{check.n_blocks} complete blocks over {check.n_clips} clips, unit `{check.unit}`, "
            f"V0 mean error {cell_text(check.v0)}. Incomplete blocks: {incomplete}. "
            f"Dropped bootstrap replicas: {check.bootstrap_dropped}."
        ),
        "",
    ]
    lines += text_table(("cell", "mean abs error", *bounds), _named(check.cells, check.cells_ci))
    lines += text_table(("effect", "value", *bounds), _named(check.effects, check.effects_ci))
    lines += text_table(("stage", "Shapley", *bounds), _named(check.shapley, check.shapley_ci))
    lines += text_table(("budget row", "value", "%", *bounds, "Level 1 metric"), budget)
    return section(f"{check.check_id}", lines)


@_body.register
def _attribution_body(report: AttributionReport) -> list[str]:
    lines = _header("Error attribution by stage (Level 3)", report.provenance)
    fixed = ", ".join(stage.value for stage in report.fixed_ground_truth)
    lines += [
        (
            f"Report split `{report.report_split.value}`. Fixed ground truth: {fixed}. "
            f"Clip bootstrap: {report.bootstrap} replicas, seed {report.seed}, "
            f"{cell_text(100 * report.ci)}% intervals. "
            f"LIFT unit in every cell: `{report.lift_unit}`."
        ),
        "",
    ]
    for check in report.checks:
        lines += _attribution(check, report.ci)
    return lines + _failures(report.failures)


@_body.register
def _real_body(report: RealVideoReport) -> list[str]:
    lines = ["# Real video diagnosis (end-to-end only)", "", REAL_VIDEO_SCOPE, ""]
    lines += _provenance(report.provenance)
    lines += motion_section(report.motion)
    lines += fall_section(report.falls)
    lines += verdict_section(report.verdicts, report.thresholds_provisional)
    return lines + _failures(report.failures)
