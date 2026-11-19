"""Shapley contribution of each perception stage to a check's error against the oracle, with its
bootstrap interval, one panel per check that has attributed clips (in the check's own unit)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from physverify.diagnosis.report import AttributionReport, CheckAttribution
from physverify.reporting.figures._base import NoDataError, Output, Value
from physverify.reporting.figures._render import (
    bars,
    category_rows,
    grid,
    numbers,
    series,
    with_intervals,
)
from physverify.schema.stages import Stage

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure as Canvas

    from physverify.reporting.figures._labels import Text
    from physverify.reporting.figures._records import Records

COLUMNS = 3
PANEL_TICK_INTERVALS = 3
ROW_HEIGHT = 1.45


@dataclass(frozen=True)
class StageShare:
    stage: str
    share: Value


@dataclass(frozen=True)
class CheckShares:
    check_id: str
    unit: str
    n_clips: int
    stages: tuple[StageShare, ...]
    """In pipeline order."""


def stage_shares(report: AttributionReport) -> tuple[CheckShares, ...]:
    found = tuple(_shares(c) for c in report.checks if c.n_clips > 0 and c.shapley)
    if not found:
        raise NoDataError("no check of the attribution has attributed clips")
    return found


def _shares(check: CheckAttribution) -> CheckShares:
    stages = [stage for stage in Stage if stage in check.shapley]
    return CheckShares(
        check.check_id,
        check.unit,
        check.n_clips,
        tuple(StageShare(stage.value, _share(check, stage)) for stage in stages),
    )


def _share(check: CheckAttribution, stage: Stage) -> Value:
    interval = check.shapley_ci.get(stage)
    low, high = (math.nan, math.nan) if interval is None else (interval.low, interval.high)
    return Value.finite(check.shapley[stage], low, high)


class StageAttribution:
    name: ClassVar[str] = "stage_attribution"
    inputs: ClassVar[tuple[Output, ...]] = (Output.ATTRIBUTION,)
    texts: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "x_label": "Shapley contribution to the check's error, in its unit",
            "panel": "{check} (n = {n} clips)",
            "missing": "not estimated",
        }
    )

    def data(self, records: Records) -> tuple[CheckShares, ...]:
        return stage_shares(records.attribution())

    def draw(self, data: tuple[CheckShares, ...], text: Text) -> Canvas:
        found = {share.stage for check in data for share in check.stages}
        stages: list[str] = [stage.value for stage in Stage if stage.value in found]
        rows = {stage: row for row, stage in enumerate(stages)}
        canvas, axes = grid(math.ceil(len(data) / COLUMNS), COLUMNS, ROW_HEIGHT, sharey=True)
        flat: list[Any] = list(axes.flat)
        for ax, check in zip(flat, data, strict=False):
            _draw_panel(ax, check, rows, text)
        for ax in flat[len(data) :]:
            ax.set_axis_off()
        category_rows(flat[0], [text.name("stages", s) for s in stages])
        canvas.supxlabel(text("x_label"))
        return canvas


def _draw_panel(ax: Axes, check: CheckShares, rows: Mapping[str, int], text: Text) -> None:
    places = [rows[stage.stage] for stage in check.stages]
    shares = [stage.share for stage in check.stages]
    bars(ax, places, [s.value for s in shares], text("missing"))
    with_intervals(ax, places, shares, series(0), "", horizontal=True, line=False)
    ax.axvline(0.0, color="0.0", linewidth=0.6)
    numbers(ax.xaxis, text, most=PANEL_TICK_INTERVALS)
    ax.set_xlabel(text.name("units", check.unit))
    check_name = text.name("checks", check.check_id)
    ax.set_title(text("panel", check=check_name, n=check.n_clips), loc="left")


PLUGINS = (StageAttribution,)
