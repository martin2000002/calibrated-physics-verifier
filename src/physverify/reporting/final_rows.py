"""The rows of the final table: status, thresholds, the row record and the shared readers."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from physverify.calibration.artifact import EvaluationReport, EvaluationStratum
from physverify.core.jsonio import read_json

Status = Literal["pass", "fail", "not_evaluable", "descriptive"]
PASS: Status = "pass"
FAIL: Status = "fail"
NOT_EVALUABLE: Status = "not_evaluable"
DESCRIPTIVE: Status = "descriptive"
"""Printed next to a criterion, judged by none."""
OUTSIDE_RUN = "checked at the merge to main, outside the final run"


@dataclass(frozen=True)
class Criteria:
    """The thresholds of the closing criteria."""

    oracle_met: int = 9
    perceived_goal: int = 6
    perceived_minimum: int = 5
    perceived_auroc: float = 0.70
    real_fpr: float = 0.10
    real_fpr_upper: float = 0.20
    real_deciders: tuple[str, ...] = ("R1", "R2", "R4_jump", "R4_missing", "R4_size")
    generated_decided: float = 0.25
    kubric_decided: float = 0.60
    risk_target: float = 0.10
    max_missing_input: float = 0.05


@dataclass(frozen=True)
class Row:
    id: str
    criterion: str
    value: str
    ci: str
    threshold: str
    status: Status
    source: str
    note: str = ""


def fmt(value: float | None) -> str:
    return "n/a" if value is None or value != value else f"{value:.3f}"


def ci_text(low: float | None, high: float | None) -> str:
    return "" if low is None and high is None else f"[{fmt(low)}, {fmt(high)}]"


def missing(id_: str, criterion: str, threshold: str, what: str) -> Row:
    return Row(id_, criterion, "n/a", "", threshold, NOT_EVALUABLE, "", f"no {what}")


def load_evaluation(path: Path) -> EvaluationReport:
    return EvaluationReport.model_validate(read_json(path))


def stratum_at(report: EvaluationReport, level: str) -> EvaluationStratum | None:
    return next((s for s in report.strata if s.level.value == level), None)


def counted(stratum: EvaluationStratum) -> tuple[str, ...]:
    return tuple(sorted(c.check_id for c in stratum.heldout_h1 if c.counted))


def input_gaps_over(report: EvaluationReport, active: Collection[str], limit: float) -> list[str]:
    """The active checks of a set left without input on more than ``limit`` of its clips."""
    return [
        f"{g.dataset} {g.level.value} {g.check_id} {g.missing}/{g.n}"
        for g in report.missing_input
        if g.check_id in active and g.n and g.missing / g.n > limit
    ]
