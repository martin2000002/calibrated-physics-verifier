"""Share of clips flagged per injected level, oracle and perceived inputs, one panel per check.

The evaluation keeps no per-clip score, so the panels show verdict rates, not a score against its
threshold."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from physverify.calibration.artifact import EvaluationReport, HeldoutCheck
from physverify.reporting.figures._base import NoDataError, Output, Value
from physverify.reporting.figures._render import (
    grid,
    numbers,
    outside_legend,
    stratum_series,
    with_intervals,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure as Canvas

    from physverify.reporting.figures._labels import Text
    from physverify.reporting.figures._records import Records

COLUMNS = 3
ROW_HEIGHT = 1.55
DODGE = 0.1
MAGNITUDE_DIGITS = 3


@dataclass(frozen=True)
class Curve:
    stratum: str
    rates: tuple[Value, ...]
    """The FPR on clean clips, then the detection rate of each of the panel's levels (an
    abstention counts as a miss; no value where this input level lacks the level)."""


@dataclass(frozen=True)
class Panel:
    check_id: str
    magnitudes: tuple[float | None, ...]
    """Median injected magnitude of each level id any input level reports, in id order."""
    curves: tuple[Curve, ...]


def detection_panels(report: EvaluationReport) -> tuple[Panel, ...]:
    by_stratum = {s.level.value: {c.check_id: c for c in s.checks} for s in report.strata}
    order = dict.fromkeys(check for checks in by_stratum.values() for check in checks)
    if not order:
        raise NoDataError("the evaluation reports no held-out check")
    return tuple(_panel(check_id, by_stratum) for check_id in order)


def _panel(check_id: str, by_stratum: Mapping[str, Mapping[str, HeldoutCheck]]) -> Panel:
    found = {name: checks[check_id] for name, checks in by_stratum.items() if check_id in checks}
    magnitudes = _magnitudes(found.values())
    levels = sorted(magnitudes)
    return Panel(
        check_id,
        tuple(magnitudes[level] for level in levels),
        tuple(Curve(name, _rates(check, levels)) for name, check in found.items()),
    )


def _magnitudes(checks: Iterable[HeldoutCheck]) -> dict[int, float | None]:
    """Level id -> its median magnitude, from the first input level that reports one."""
    found: dict[int, float | None] = {}
    for check in checks:
        for level in check.tpr_by_level:
            if found.get(level.level) is None:
                found[level.level] = level.mu_median
    return found


def _rates(check: HeldoutCheck, levels: Sequence[int]) -> tuple[Value, ...]:
    """Aligned on ``levels`` by id: a level this input level lacks has no value."""
    by_id = {level.level: Value.of_rate(level.total) for level in check.tpr_by_level}
    return (Value.of_rate(check.fpr_clean), *(by_id.get(level, Value(None)) for level in levels))


class CheckDetection:
    name: ClassVar[str] = "check_detection"
    inputs: ClassVar[tuple[Output, ...]] = (Output.KUBRIC_EVAL,)
    texts: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "x_label": "Injected magnitude (median of each level)",
            "y_label": "Share of clips flagged",
            "clean": "clean",
        }
    )

    def data(self, records: Records) -> tuple[Panel, ...]:
        return detection_panels(records.kubric())

    def draw(self, data: tuple[Panel, ...], text: Text) -> Canvas:
        rows = math.ceil(len(data) / COLUMNS)
        canvas, axes = grid(rows, COLUMNS, ROW_HEIGHT, sharey=True)
        flat: list[Any] = list(axes.flat)
        for ax, panel in zip(flat, data, strict=False):
            _draw_panel(ax, panel, text)
        for ax in flat[len(data) :]:
            ax.set_axis_off()
        canvas.supxlabel(text("x_label"))
        canvas.supylabel(text("y_label"))
        outside_legend(canvas, flat[: len(data)], loc="outside upper center")
        return canvas


def _draw_panel(ax: Axes, panel: Panel, text: Text) -> None:
    for index, curve in enumerate(panel.curves):
        offset = DODGE * (index - (len(panel.curves) - 1) / 2)
        positions = [step + offset for step in range(len(curve.rates))]
        with_intervals(
            ax, positions, curve.rates, stratum_series(curve.stratum), text(curve.stratum)
        )
    ticks = [
        text("clean"),
        *("" if m is None else text.tick(m, MAGNITUDE_DIGITS) for m in panel.magnitudes),
    ]
    ax.set_xticks(range(len(ticks)), ticks)
    ax.set_ylim(-0.04, 1.04)
    numbers(ax.yaxis, text)
    ax.set_title(text.name("checks", panel.check_id), loc="left")


PLUGINS = (CheckDetection,)
