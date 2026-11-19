"""False alarms on real video, per check and real set (perceived inputs): the FPR with its
interval against the criterion's limits, and the share of clips each check decides."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from physverify.calibration.artifact import EvaluationReport, RealFpr
from physverify.reporting.figures._base import NoDataError, Output, Value
from physverify.reporting.figures._render import (
    bars,
    category_rows,
    grid,
    numbers,
    outside_legend,
    reference,
    series,
    with_intervals,
)
from physverify.reporting.final_rows import Criteria, stratum_at
from physverify.schema.checks import InputLevel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure as Canvas

    from physverify.reporting.figures._labels import Text
    from physverify.reporting.figures._records import Records

CHECK_HEIGHT = 0.17
"""Inches per check in each row of panels: rates above, decided shares below."""
FRAME = 1.2
"""Inches for the titles, the axis labels and the legend."""
DECIDED, OFF = series(0), series(3)


@dataclass(frozen=True)
class CheckAlarms:
    check_id: str
    fpr: Value
    decided: int
    excluded: bool


@dataclass(frozen=True)
class RealSet:
    dataset: str
    n_clips: int | None
    """None when the evaluation does not list the set."""
    checks: tuple[CheckAlarms, ...]

    def decided_share(self, check: CheckAlarms) -> float | None:
        return check.decided / self.n_clips if self.n_clips else None


@dataclass(frozen=True)
class RealAlarms:
    sets: tuple[RealSet, ...]
    limit: float
    upper: float
    """The limit of the interval's upper bound."""


def real_alarms(
    report: EvaluationReport, excluded: frozenset[str], criteria: Criteria
) -> RealAlarms:
    perceived = stratum_at(report, InputLevel.PERCEIVED.value)
    if perceived is None or not perceived.real:
        raise NoDataError("the evaluation has no real-video rows with perceived inputs")
    clips = {ref.name: ref.n_clips for ref in report.real_sets}
    names = dict.fromkeys([*clips, *(row.dataset for row in perceived.real)])
    sets = (_set(name, clips.get(name), perceived.real, excluded) for name in names)
    return RealAlarms(
        tuple(s for s in sets if s.checks), criteria.real_fpr, criteria.real_fpr_upper
    )


def _set(
    name: str, n_clips: int | None, rows: tuple[RealFpr, ...], excluded: frozenset[str]
) -> RealSet:
    checks = tuple(
        CheckAlarms(row.check_id, Value.of_rate(row.fpr), row.fpr.n, row.check_id in excluded)
        for row in rows
        if row.dataset == name
    )
    return RealSet(name, n_clips, checks)


class RealFalseAlarms:
    name: ClassVar[str] = "real_false_alarms"
    inputs: ClassVar[tuple[Output, ...]] = (Output.REAL_EVAL,)
    texts: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "fpr_label": "False alarm rate",
            "decided_label": "Share decided",
            "set": "{dataset} (n = {n} clips)",
            "judged": "Active check",
            "excluded": "Excluded check",
            "limit": "FPR limit ({value})",
            "upper": "Limit of the upper bound ({value})",
            "unknown": "no clip count",
        }
    )

    def data(self, records: Records) -> RealAlarms:
        return real_alarms(records.real(), records.excluded, records.criteria)

    def draw(self, data: RealAlarms, text: Text) -> Canvas:
        checks = list(dict.fromkeys(c.check_id for real in data.sets for c in real.checks))
        row_height = CHECK_HEIGHT * len(checks) + FRAME / 2
        canvas, axes = grid(2, len(data.sets), row_height, sharex="row", sharey=True)
        rows = {check: row for row, check in enumerate(checks)}
        for column, real in enumerate(data.sets):
            _rates(axes[0, column], real, rows, data, text)
            _decided(axes[1, column], real, rows, text)
        category_rows(axes[0, 0], [text.name("checks", c) for c in checks])
        outside_legend(canvas, list(axes[0, :]))
        return canvas


def _rates(ax: Axes, real: RealSet, rows: Mapping[str, int], data: RealAlarms, text: Text) -> None:
    for check in real.checks:
        style, key = (OFF, "excluded") if check.excluded else (DECIDED, "judged")
        with_intervals(
            ax, [rows[check.check_id]], [check.fpr], style, text(key), horizontal=True, line=False
        )
    reference(ax, data.limit, text("limit", value=text.number(data.limit)), vertical=True)
    upper = text("upper", value=text.number(data.upper))
    reference(ax, data.upper, upper, vertical=True, dash=":")
    ax.set_xlim(-0.02, 1.02)
    numbers(ax.xaxis, text)
    ax.set_xlabel(text("fpr_label"))
    name = text.name("datasets", real.dataset)
    title = name if real.n_clips is None else text("set", dataset=name, n=real.n_clips)
    ax.set_title(title, loc="left")


def _decided(ax: Axes, real: RealSet, rows: Mapping[str, int], text: Text) -> None:
    places = [rows[check.check_id] for check in real.checks]
    bars(ax, places, [real.decided_share(check) for check in real.checks], text("unknown"))
    ax.set_xlim(0.0, 1.0)
    numbers(ax.xaxis, text)
    ax.set_xlabel(text("decided_label"))


PLUGINS = (RealFalseAlarms,)
