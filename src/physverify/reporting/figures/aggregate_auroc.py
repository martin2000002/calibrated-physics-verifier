"""AUROC of the clip scores, perturbed against clean held-out clips, per input level.

B1 (the verifier's graded score) is judged against the criterion; 1 - S is descriptive. The
evaluation keeps the area and its interval, not the ROC curve."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from physverify.calibration.artifact import Auroc, EvaluationReport, EvaluationStratum
from physverify.reporting.figures._base import CHANCE, NoDataError, Output, Value
from physverify.reporting.figures._render import (
    grid,
    numbers,
    outside_legend,
    reference,
    stratum_series,
    with_intervals,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure as Canvas

    from physverify.reporting.figures._labels import Text
    from physverify.reporting.figures._records import Records

B1, ONE_MINUS_S = "b1", "one_minus_s"
ROW_HEIGHT = 0.32
MARGIN = 0.9
LOWEST = 0.4


@dataclass(frozen=True)
class AurocRow:
    score: str
    stratum: str
    auroc: Value
    in_sample: bool


@dataclass(frozen=True)
class AurocData:
    rows: tuple[AurocRow, ...]
    target: float
    """The criterion of perceived B1."""


def auroc_rows(report: EvaluationReport, target: float) -> AurocData:
    rows = tuple(row for stratum in report.strata for row in _rows(stratum))
    if not rows:
        raise NoDataError("the evaluation reports no AUROC")
    return AurocData(rows, target)


def _rows(stratum: EvaluationStratum) -> list[AurocRow]:
    scores: tuple[tuple[str, Auroc | None], ...] = (
        (B1, stratum.auroc_graded),
        (ONE_MINUS_S, stratum.auroc),
    )
    return [
        AurocRow(score, stratum.level.value, Value(a.value, a.ci.low, a.ci.high), a.in_sample)
        for score, a in scores
        if a is not None and a.value is not None
    ]


class AggregateAuroc:
    name: ClassVar[str] = "aggregate_auroc"
    inputs: ClassVar[tuple[Output, ...]] = (Output.KUBRIC_EVAL,)
    texts: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "x_label": "AUROC, perturbed against clean held-out clips",
            B1: "B1",
            ONE_MINUS_S: "1 \N{MINUS SIGN} S (descriptive)",
            "row": "{score}, {stratum}",
            "in_sample": "{row} (in sample)",
            "chance": "Chance",
            "target": "Criterion for perceived B1 ({value})",
        }
    )

    def data(self, records: Records) -> AurocData:
        return auroc_rows(records.kubric(), records.criteria.perceived_auroc)

    def draw(self, data: AurocData, text: Text) -> Canvas:
        canvas, axes = grid(1, 1, MARGIN + ROW_HEIGHT * len(data.rows))
        ax = axes[0, 0]
        for position, row in enumerate(data.rows):
            style = stratum_series(row.stratum)
            with_intervals(ax, [position], [row.auroc], style, "", horizontal=True, line=False)
        ax.set_yticks(range(len(data.rows)), [_row_label(row, text) for row in data.rows])
        ax.invert_yaxis()
        reference(ax, CHANCE, text("chance"), vertical=True, dash=":")
        reference(ax, data.target, text("target", value=text.number(data.target)), vertical=True)
        lows = [row.auroc.low for row in data.rows if row.auroc.low is not None]
        ax.set_xlim(min([LOWEST, *lows]) - 0.02, 1.0)
        numbers(ax.xaxis, text)
        ax.set_xlabel(text("x_label"))
        outside_legend(canvas, [ax])
        return canvas


def _row_label(row: AurocRow, text: Text) -> str:
    label = text("row", score=text(row.score), stratum=text(row.stratum))
    return text("in_sample", row=label) if row.in_sample else label


PLUGINS = (AggregateAuroc,)
