"""Human-readable sheets of the artifact and the evaluation: Markdown plus ``curves.csv``."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence

from physverify.calibration.schema import (
    CalibrationArtifact,
    CheckCalibration,
    Ci,
    DatasetRef,
    DetectionLimit,
    EvaluationReport,
    EvaluationStratum,
    H1PrimaryView,
    HeldoutCheck,
    Rate,
    Stratum,
)

CALIBRATION_MD = "calibration.md"
CURVES_CSV = "curves.csv"
EVALUATION_MD = "evaluation.md"
CURVE_COLUMNS = (
    "level",
    "profile",
    "check_id",
    "pert_level",
    "mu_median",
    "tpr_total",
    "tpr_low",
    "tpr_high",
    "tpr_decided",
    "n",
)
ESTIMATE_NOTE = (
    "Estimates conditional on the calib design (Kubric). p_c and S are not real-world "
    "probabilities; perceived sigma factors are measured, not exact."
)


def _num(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}g}"


def _ci(ci: Ci) -> str:
    return f"[{_num(ci.low)}, {_num(ci.high)}]"


def _rate(r: Rate) -> str:
    text = f"{_num(r.value)} {_ci(r.ci)} ({r.k}/{r.n})"
    if r.ci_method != "clopper_pearson":
        text += f" {r.n_scenes} scenes, {r.ci_method}"
    return text


def _ld(ld: DetectionLimit) -> str:
    opened = f" open {','.join(ld.ci_open)}" if ld.ci_open else ""
    reason = f" {ld.reason}" if ld.reason else ""
    return f"{_num(ld.value)} {_ci(ld.ci)} {ld.censored}{opened}{reason} (estimate)"


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _check_row(check_id: str, c: CheckCalibration) -> list[str]:
    ld = c.detection_limit
    return [
        check_id,
        c.status,
        _num(c.tau),
        c.unit,
        str(c.n_clean_scenes) + (f" ({c.n_clean_dropped} NaN)" if c.n_clean_dropped else ""),
        f"{_num(c.tau_conformal)} / {_num(c.tau_tolerance)}",
        _rate(c.fpr.cv),
        _rate(c.fpr.cv_decided),
        _rate(c.fpr.off_target) + (" coupling?" if c.fpr.undeclared_coupling else ""),
        f"{_num(c.auroc.value)} {_ci(c.auroc.ci)}",
        _ld(ld),
        f"{_num(c.monotonicity.kendall_tau_b)} {_ci(c.monotonicity.ci)}",
        _num(c.weight.value),
        _num(c.sigma.factor),
    ]


CHECK_HEADERS = (
    "check",
    "status",
    "tau",
    "unit",
    "clean scenes",
    "q / tolerance",
    "FPR cv",
    "FPR cv (decided)",
    "FPR off-target",
    "AUROC (in-sample)",
    "L_D",
    "tau_b",
    "weight",
    "sigma factor",
)


def _stratum_lines(s: Stratum) -> list[str]:
    agg = s.aggregate
    lines = [f"## {s.level.value} / {s.profile} (row_step {s.row_step}, k_sigma {s.k_sigma})", ""]
    lines += _table(CHECK_HEADERS, (_check_row(c, s.checks[c]) for c in sorted(s.checks)))
    lines += [
        "",
        f"Score threshold s* = {_num(agg.score_threshold)}"
        f"{' (in-sample, optimistic)' if agg.score_threshold_in_sample else ''} "
        f"(sufficient: {agg.sufficient}, "
        f"{agg.n_clean_scenes} clean / {agg.n_perturbed_clips} perturbed clips); "
        f"AUROC {_num(agg.auroc.value)} {_ci(agg.auroc.ci)}.",
        "",
    ]
    return lines


def calibration_markdown(artifact: CalibrationArtifact) -> str:
    d = artifact.dataset
    lines = [
        f"# Calibration {artifact.run_id} ({artifact.status})",
        "",
        f"Dataset {d.name} {d.version} subset {d.subset} split {d.split}: {d.n_clips} clips, "
        f"{d.n_scenes} scenes. Commit {artifact.commit}; alpha {artifact.settings.alpha}, "
        f"beta {artifact.settings.beta}, ci {artifact.settings.ci}.",
        "",
        ESTIMATE_NOTE,
        "",
    ]
    for stratum in artifact.strata:
        lines += _stratum_lines(stratum)
    return "\n".join(lines)


def curves_csv(artifact: CalibrationArtifact) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CURVE_COLUMNS)
    for s in artifact.strata:
        for check_id in sorted(s.checks):
            for t in s.checks[check_id].tpr_by_level:
                writer.writerow(
                    [
                        s.level.value,
                        s.profile,
                        check_id,
                        t.level,
                        _num(t.mu_median, 6),
                        _num(t.total.value, 6),
                        _num(t.total.ci.low, 6),
                        _num(t.total.ci.high, 6),
                        _num(t.decided.value, 6),
                        t.total.n,
                    ]
                )
    return buffer.getvalue()


def _evaluation_stratum(s: EvaluationStratum) -> list[str]:
    lines = [f"## {s.level.value} / {s.profile}", ""]
    lines += _table(
        ("check", "fit status", "tau", "FPR clean", "FPR clean (decided)", "FPR off-target"),
        (
            [
                c.check_id,
                c.status,
                _num(c.tau),
                _rate(c.fpr_clean),
                _rate(c.fpr_clean_decided),
                _rate(c.fpr_off_target),
            ]
            for c in s.checks
        ),
    )
    if s.scene_kendall:
        lines += ["", "Per-scene Kendall tau_b over levels:", ""]
        lines += _table(
            ("scene", "check", "tau_b", "levels", "split"),
            (
                [k.scene_id, k.check_id, _num(k.kendall_tau_b), str(k.n_levels), k.split]
                for k in s.scene_kendall
            ),
        )
    if s.real or s.real_clip_flags:
        lines += ["", "Real video (false-positive control):", ""]
        lines += _table(
            ("dataset", "check", "FPR"),
            ([r.dataset, r.check_id, _rate(r.fpr)] for r in (*s.real, *s.real_clip_flags)),
        )
    return [*lines, ""]


def evaluation_markdown(report: EvaluationReport) -> str:
    lines = [
        f"# Evaluation {report.run_id}",
        "",
        f"Artifact {report.artifact_path} (sha256 {report.artifact_sha256[:16]}, "
        f"run {report.artifact_run_id}); dataset {report.dataset.name} subset "
        f"{report.dataset.subset}: {report.dataset.n_clips} clips.",
        "",
        ESTIMATE_NOTE,
        "",
    ]
    for stratum in report.strata:
        lines += _evaluation_stratum(stratum)
    if report.h1_primary is not None:
        lines += _h1_primary(report.h1_primary, report.strata, report.dataset)
    return "\n".join(lines)


def _h1_primary(
    view: H1PrimaryView, everything: Sequence[EvaluationStratum], dataset: DatasetRef
) -> list[str]:
    """Side by side: the primary scenes (ADR 0015 1.4) against every scene."""
    lines = [
        f"## H1 primary (scenes {', '.join(view.scenes)}) vs all scenes",
        "",
        f"Primary: {view.dataset.n_clips} clips, {view.dataset.n_scenes} scenes. "
        f"All: {dataset.n_clips} clips, {dataset.n_scenes} scenes.",
        "",
    ]
    for primary in view.strata:
        whole = next(
            (s for s in everything if (s.level, s.profile) == (primary.level, primary.profile)),
            None,
        )
        if whole is not None:
            lines += _side_by_side(primary, whole)
    return lines


def _side_by_side(primary: EvaluationStratum, whole: EvaluationStratum) -> list[str]:
    by_id = {c.check_id: c for c in whole.checks}
    lines = [f"### {primary.level.value} / {primary.profile}", ""]
    lines += _table(
        (
            "check",
            "tau",
            "FPR clean all",
            "FPR clean primary",
            "FPR off-target all",
            "FPR off-target primary",
            "TPR top level all",
            "TPR top level primary",
        ),
        (
            [
                c.check_id,
                _num(c.tau),
                _rate(by_id[c.check_id].fpr_clean),
                _rate(c.fpr_clean),
                _rate(by_id[c.check_id].fpr_off_target),
                _rate(c.fpr_off_target),
                _top_tpr(by_id[c.check_id]),
                _top_tpr(c),
            ]
            for c in primary.checks
            if c.check_id in by_id
        ),
    )
    return [*lines, ""]


def _top_tpr(check: HeldoutCheck) -> str:
    if not check.tpr_by_level:
        return "-"
    top = max(check.tpr_by_level, key=lambda t: t.level)
    return f"L{top.level}: {_rate(top.total)}"
