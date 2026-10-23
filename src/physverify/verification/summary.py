"""The level a ``verify`` run reads and its console summary."""

from __future__ import annotations

from collections.abc import Sequence

from physverify.core.errors import UsageError
from physverify.core.markdown import number
from physverify.schema.checks import InputLevel, Verdict
from physverify.verification.runner import RowStatus, VerificationReport
from physverify.verification.sweep import MatrixCell, Rule, RuleOutcome, SpecificityReport

LEVELS = (InputLevel.ORACLE.value, InputLevel.PERCEIVED.value)


def parse_level(level: str | None, gt: Sequence[str], sweep: bool) -> InputLevel:
    """ORACLE, PERCEIVED, or MIXED when a perceived run replaces stages with ground truth."""
    if level not in LEVELS:
        raise UsageError(f"--level is required: one of {list(LEVELS)}")
    parsed = InputLevel(level)
    if parsed is InputLevel.ORACLE and gt:
        raise UsageError("--gt applies to --level perceived; the oracle is ground truth already")
    if sweep and parsed is not InputLevel.ORACLE:
        raise UsageError(
            "--sweep needs --level oracle (synthetic labels are exact only on the oracle)"
        )
    return InputLevel.MIXED if gt else parsed


def summary_lines(report: VerificationReport) -> list[str]:
    lines: list[str] = []
    check_ids = list(dict.fromkeys(row.check_id for row in report.table))
    for check_id in check_ids:
        rows = [row for row in report.table if row.check_id == check_id]
        counts = {status: sum(r.status is status for r in rows) for status in RowStatus}
        fails = sum(r.verdict is Verdict.FAIL for r in rows)
        tally = " ".join(f"{status.value} {n}" for status, n in counts.items())
        lines.append(f"{check_id}: {tally} FAIL {fails}")
    return lines


def matrix_lines(cells: Sequence[MatrixCell]) -> list[str]:
    lines = ["sweep: perturbation check median max n/clips"]
    for cell in cells:
        lines.append(
            f"{cell.perturbation} {cell.check_id} {_number(cell.median)} {_number(cell.max)} "
            f"{cell.n}/{cell.n_clips}"
        )
    return lines


def specificity_lines(spec: SpecificityReport) -> list[str]:
    verdict = "PASS" if spec.passed else "FAIL"
    lines = [f"specificity: {verdict} ({len(spec.outcomes)} outcomes, {len(spec.failures)} failed)"]
    for outcome in spec.outcomes:
        if outcome.passed and outcome.rule is not Rule.COUPLING:
            continue
        label = _label(outcome)
        lines.append(
            f"{label} {outcome.rule.value} {outcome.perturbation} {outcome.check_id}: "
            f"expected {_number(outcome.expected)} observed {_number(outcome.observed)} "
            f"{outcome.detail}".rstrip()
        )
    return lines


def _label(outcome: RuleOutcome) -> str:
    if outcome.rule is Rule.COUPLING:
        return "expected coupling"
    return "UNDEFINED" if outcome.undefined else "FAILED"


def _number(value: float | None) -> str:
    return number(value, ".4g", "-", "nan")
