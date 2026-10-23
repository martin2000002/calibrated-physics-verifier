"""The console lines of ``calibrate show``, ``dev-score`` and ``stage-sigma``."""

from __future__ import annotations

from collections.abc import Sequence

from physverify.calibration.artifact import StageFactor
from physverify.calibration.artifact_io import Calibration
from physverify.calibration.dev_score import DevScoreReport, LevelDevScore
from physverify.core.markdown import number


def show_lines(
    calibration: Calibration, level: str | None = None, profile: str | None = None
) -> list[str]:
    lines = [f"calibration {calibration.run_id} v{calibration.version} {calibration.status}"]
    for lv, prof in calibration.strata():
        if (level and lv.value != level) or (profile and prof != profile):
            continue
        stratum = calibration.stratum(lv, prof)
        lines.append(f"{lv.value}/{prof}: s* {_num(stratum.score_threshold())}")
        for check_id, c in sorted(stratum.model.checks.items()):
            lines.append(
                f"  {check_id} {c.status} tau {_num(c.tau)} {c.unit} clean {c.n_clean_scenes} "
                f"FPR cv {_num(c.fpr.cv.value)} "
                f"[{_num(c.fpr.cv.ci.low)}, {_num(c.fpr.cv.ci.high)}] "
                f"w {_num(c.weight.value)}"
            )
    return lines


def devscore_lines(report: DevScoreReport) -> list[str]:
    lines = [f"note: {report.scope_note}"]
    for level in report.levels:
        lines += _level_lines(level)
    chosen = report.selection
    if chosen is not None:
        evaluable = "-" if chosen.oracle_evaluable is None else ",".join(chosen.oracle_evaluable)
        lines.append(
            f"selection ({chosen.level.value}): n_met {chosen.n_met}; oracle evaluable "
            f"[{evaluable}]; all met {chosen.all_oracle_evaluable_met}; "
            f"tie-break AUROC_cv {_num(chosen.tiebreak_auroc_cv)}"
        )
    return lines


def _level_lines(level: LevelDevScore) -> list[str]:
    auroc = level.auroc_cv
    lines = [
        (
            f"{level.level.value}/{level.profile}: {level.n_met}/{len(level.checks)} meet "
            f"monotonicity and J_cv^lo > 0 ({','.join(level.met) or 'none'}); AUROC_cv "
            f"{_num(auroc.value)} [{_num(auroc.ci.low)}, {_num(auroc.ci.high)}]"
        )
    ]
    for check_id, c in sorted(level.checks.items()):
        lines.append(
            f"  {check_id}: tau_b {_num(c.h1.kendall_tau_b)} [{_num(c.h1.ci.low)}, "
            f"{_num(c.h1.ci.high)}] monotone {c.meets_h1}; J_cv {_num(c.j_cv.informedness)} "
            f"[{_num(c.j_cv.informedness_ci.low)}, {_num(c.j_cv.informedness_ci.high)}]"
        )
    return lines


def factor_lines(factors: Sequence[StageFactor]) -> list[str]:
    """One line per stage factor: the coverage at k = 1 before and (out of fold) after."""
    lines = []
    for f in factors:
        before = next((c.observed for c in f.coverage_before if c.k == 1.0), None)
        after = next((c.observed for c in f.coverage_after if c.k == 1.0), None)
        lines.append(
            f"{f.stage}/{f.provider} {f.quantity}: factor {_num(f.factor)} n {f.n} "
            f"coverage@1 before {_num(before)} after (out of fold) {_num(after)}"
        )
    return lines


def _num(value: float | None) -> str:
    return number(value, ".4g", "-")
