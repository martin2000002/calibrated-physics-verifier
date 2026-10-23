"""Human-readable sheets of the artifact and the evaluation: Markdown plus ``curves.csv``."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence

from physverify.calibration.heldout_h1 import n_met
from physverify.calibration.schema import (
    CalibrationArtifact,
    CheckCalibration,
    CheckMonotonicity,
    Ci,
    DatasetRef,
    DetectionLimit,
    EvaluationReport,
    EvaluationStratum,
    H1PrimaryView,
    HeldoutCheck,
    HeldoutH1Check,
    Monotonicity,
    Rate,
    Stratum,
)
from physverify.core.markdown import number, table

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


def number_text(value: float | None, digits: int = 3) -> str:
    return number(value, f".{digits}g", "-")


def ci_text(ci: Ci) -> str:
    return f"[{number_text(ci.low)}, {number_text(ci.high)}]"


def rate_text(r: Rate) -> str:
    text = f"{number_text(r.value)} {ci_text(r.ci)} ({r.k}/{r.n})"
    if r.ci_method != "clopper_pearson":
        text += f" {r.n_scenes} scenes, {r.ci_method}"
    return text


def detection_limit_text(ld: DetectionLimit) -> str:
    opened = f" open {','.join(ld.ci_open)}" if ld.ci_open else ""
    reason = f" {ld.reason}" if ld.reason else ""
    return f"{number_text(ld.value)} {ci_text(ld.ci)} {ld.censored}{opened}{reason} (estimate)"


def _check_row(check_id: str, c: CheckCalibration) -> list[str]:
    ld = c.detection_limit
    return [
        check_id,
        c.status,
        number_text(c.tau),
        c.unit,
        str(c.n_clean_scenes) + (f" ({c.n_clean_dropped} NaN)" if c.n_clean_dropped else ""),
        f"{number_text(c.tau_conformal)} / {number_text(c.tau_tolerance)}",
        rate_text(c.fpr.cv),
        rate_text(c.fpr.cv_decided),
        rate_text(c.fpr.off_target) + (" coupling?" if c.fpr.undeclared_coupling else ""),
        f"{number_text(c.auroc.value)} {ci_text(c.auroc.ci)}",
        detection_limit_text(ld),
        f"{number_text(c.monotonicity.kendall_tau_b)} {ci_text(c.monotonicity.ci)}",
        number_text(c.weight.value),
        number_text(c.sigma.factor),
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
    lines += table(CHECK_HEADERS, (_check_row(c, s.checks[c]) for c in sorted(s.checks)))
    lines += [
        "",
        f"Score threshold s* = {number_text(agg.score_threshold)}"
        f"{' (in-sample, optimistic)' if agg.score_threshold_in_sample else ''} "
        f"(sufficient: {agg.sufficient}, "
        f"{agg.n_clean_scenes} clean / {agg.n_perturbed_clips} perturbed clips); "
        f"AUROC {number_text(agg.auroc.value)} {ci_text(agg.auroc.ci)}.",
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
                        number_text(t.mu_median, 6),
                        number_text(t.total.value, 6),
                        number_text(t.total.ci.low, 6),
                        number_text(t.total.ci.high, 6),
                        number_text(t.decided.value, 6),
                        t.total.n,
                    ]
                )
    return buffer.getvalue()


def _evaluation_stratum(s: EvaluationStratum) -> list[str]:
    lines = [f"## {s.level.value} / {s.profile}", ""]
    lines += table(
        ("check", "fit status", "tau", "FPR clean", "FPR clean (decided)", "FPR off-target"),
        (
            [
                c.check_id,
                c.status,
                number_text(c.tau),
                rate_text(c.fpr_clean),
                rate_text(c.fpr_clean_decided),
                rate_text(c.fpr_off_target),
            ]
            for c in s.checks
        ),
    )
    if s.scene_kendall:
        lines += ["", "Per-scene Kendall tau_b over levels:", ""]
        lines += table(
            ("scene", "check", "tau_b", "levels", "split"),
            (
                [k.scene_id, k.check_id, number_text(k.kendall_tau_b), str(k.n_levels), k.split]
                for k in s.scene_kendall
            ),
        )
    if s.monotonicity:
        lines += ["", "Pooled Kendall tau_b of the calib clips (D8):", ""]
        lines += table(
            ("check", "tau_b", "h1_pass", "pairs", "scenes"),
            ([m.check_id, *_tau_cells(m.monotonicity)] for m in s.monotonicity),
        )
    lines += _heldout_h1(s.heldout_h1, "every scene (secondary when a primary view follows)")
    if s.real or s.real_clip_flags:
        lines += ["", "Real video (false-positive control):", ""]
        lines += table(
            ("dataset", "check", "FPR"),
            ([r.dataset, r.check_id, rate_text(r.fpr)] for r in (*s.real, *s.real_clip_flags)),
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
            lines += _pooled_side_by_side(primary.monotonicity, whole.monotonicity)
        lines += _heldout_h1(primary.heldout_h1, f"the primary scenes ({primary.level.value})")
    return lines


def _heldout_h1(checks: Sequence[HeldoutH1Check], scope: str) -> list[str]:
    """J7: frozen tau and sigma correction, the artifact's bootstrap; met = both lower bounds
    above 0. Every check is listed, without a multiplicity correction."""
    if not checks:
        return []
    lines = [
        "",
        f"Held-out H1 over {scope}: {n_met(checks)} of {len(checks)} checks meet "
        "tau_b^lo > 0 and J^lo > 0 (no multiplicity correction):",
        "",
    ]
    lines += table(
        ("check", "tau_b", "h1_pass", "pairs", "scenes", "J", "positives", "negatives", "met"),
        (
            [
                c.check_id,
                *_tau_cells(c.monotonicity),
                f"{number_text(c.j.informedness)} {ci_text(c.j.informedness_ci)}",
                str(c.j.n_positive),
                str(c.j.n_negative),
                str(c.meets_both),
            ]
            for c in checks
        ),
    )
    return [*lines, ""]


def _tau_cells(m: Monotonicity) -> list[str]:
    return [
        f"{number_text(m.kendall_tau_b)} {ci_text(m.ci)}",
        str(m.h1_pass),
        str(m.n_pairs),
        str(m.n_scenes),
    ]


def _medians(m: Monotonicity) -> str:
    levels = ", ".join(f"L{k}: {number_text(v)}" for k, v in m.median_delta_by_level.items())
    return f"{levels} (monotone: {m.monotone_medians})"


def _pooled_side_by_side(
    primary: Sequence[CheckMonotonicity], whole: Sequence[CheckMonotonicity]
) -> list[str]:
    """h1_pass is the lower CI bound of tau_b above 0 (ADR 0009 change 4); the medians are
    descriptive and do not condition it."""
    if not primary:
        return []
    by_id = {m.check_id: m.monotonicity for m in whole}
    lines = ["Pooled Kendall tau_b of the calib clips (D8), primary vs all scenes:", ""]
    lines += table(
        (
            "check",
            "tau_b primary",
            "h1_pass primary",
            "pairs primary",
            "scenes primary",
            "tau_b all",
            "h1_pass all",
            "median delta by level, primary (descriptive)",
        ),
        (
            [
                m.check_id,
                *_tau_cells(m.monotonicity),
                *(_tau_cells(by_id[m.check_id])[:2] if m.check_id in by_id else ["-", "-"]),
                _medians(m.monotonicity),
            ]
            for m in primary
        ),
    )
    return [*lines, ""]


def _side_by_side(primary: EvaluationStratum, whole: EvaluationStratum) -> list[str]:
    by_id = {c.check_id: c for c in whole.checks}
    lines = [f"### {primary.level.value} / {primary.profile}", ""]
    lines += table(
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
                number_text(c.tau),
                rate_text(by_id[c.check_id].fpr_clean),
                rate_text(c.fpr_clean),
                rate_text(by_id[c.check_id].fpr_off_target),
                rate_text(c.fpr_off_target),
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
    return f"L{top.level}: {rate_text(top.total)}"
