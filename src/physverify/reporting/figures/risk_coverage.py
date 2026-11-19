"""Selective risk against coverage over the k grid, raw and calibrated thresholds, with the two
points of the calibrated curve that the text reports: k* chosen in calibration and the configured
k that the verdicts use. A point the record cannot place exactly is named in the legend and not
drawn. The record marks the risk as an estimate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from physverify.reporting.figures._base import NoDataError, Output, Value
from physverify.reporting.figures._render import (
    grid,
    highlight,
    numbers,
    outside_legend,
    reference,
    series,
    with_intervals,
)
from physverify.selective_risk.curve import risk_bounds
from physverify.selective_risk.records import CurveRecord, PointRow, SweptTable
from physverify.selective_risk.tables import CONFIGURED_K, FITTED_K

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure as Canvas

    from physverify.reporting.figures._labels import Text
    from physverify.reporting.figures._records import Records

CALIBRATED, RAW = "calibrated", "raw"
CALIBRATION, SUMMED, CONFIGURED = "calibration_k", "calibration_k_summed", "configured_k"
NO_K_STAR, OFF_GRID, NOT_EXACT, UNDECIDED = "no_k_star", "off_grid", "not_exact", "undecided"
HEIGHT = 2.6
BAND = "0.85"
K_DIGITS = 2


@dataclass(frozen=True)
class Operating:
    coverage: float
    risk: Value
    k_sigma: float


@dataclass(frozen=True)
class Gap:
    """Why a reported point is not drawn: the text key of the reason, and its k when known."""

    reason: str
    k_sigma: float | None = None


@dataclass(frozen=True)
class Marker:
    """A reported point of the calibrated curve; ``label`` is the text key of its legend entry."""

    label: str
    found: Operating | Gap


@dataclass(frozen=True)
class Counted:
    k_sigma: float
    n: int
    decided: int
    errors: int


@dataclass(frozen=True)
class Swept:
    table: str
    coverage: tuple[float, ...]
    risk: tuple[Value, ...]


@dataclass(frozen=True)
class RiskCoverage:
    curves: tuple[Swept, ...]
    markers: tuple[Marker, ...]
    target: float
    estimated: bool


def risk_coverage(record: CurveRecord, target: float, ci: float) -> RiskCoverage:
    """``ci``: the level of the record's risk intervals (``[abstention] ci``), for the points the
    record holds only per check."""
    tables = ((CALIBRATED, record.tables.calibrated), (RAW, record.tables.raw))
    curves = tuple(_swept(name, table) for name, table in tables)
    if not any(curve.coverage for curve in curves):
        raise NoDataError("the selective-risk curve has no finite point")
    markers = (calibration_marker(record, ci), configured_marker(record, ci))
    return RiskCoverage(curves, markers, target, record.estimated)


def _swept(name: str, table: SweptTable) -> Swept:
    finite = [p for p in table.curve if math.isfinite(p.coverage) and math.isfinite(p.risk)]
    points = sorted(finite, key=lambda p: p.coverage)
    return Swept(name, tuple(p.coverage for p in points), tuple(_risk(p) for p in points))


def _risk(point: PointRow) -> Value:
    return Value.finite(point.risk, *point.risk_ci)


def calibration_marker(record: CurveRecord, ci: float) -> Marker:
    """The calibrated curve's row at k* of calibration (``k_star_from_calib``). Calibration and
    the curve sweep the same k grid (``selective_risk.tables._grid``), so the match is exact; a
    k* off the curve is summed from the per-check rows at k* instead."""
    k = record.k_star_from_calib
    if k is None:
        return Marker(CALIBRATION, Gap(NO_K_STAR))
    row = next((p for p in record.tables.calibrated.curve if p.k_sigma == k), None)
    if row is not None:
        return Marker(CALIBRATION, _row_point(row))
    counted = summed(record, FITTED_K)
    return Marker(SUMMED, Gap(OFF_GRID, k) if counted is None else _counted_point(counted, ci))


def configured_marker(record: CurveRecord, ci: float) -> Marker:
    """The calibrated table at the configured k, which is not on the k grid: summed from the
    per-check rows, never interpolated between grid rows."""
    counted = summed(record, CONFIGURED_K)
    return Marker(CONFIGURED, Gap(NOT_EXACT) if counted is None else _counted_point(counted, ci))


def _row_point(row: PointRow) -> Operating | Gap:
    if not row.decided:
        return Gap(UNDECIDED, row.k_sigma)
    return Operating(row.coverage, _risk(row), row.k_sigma)


def _counted_point(counted: Counted, ci: float) -> Operating | Gap:
    k, n, decided, errors = counted.k_sigma, counted.n, counted.decided, counted.errors
    if not decided:
        return Gap(UNDECIDED, k)
    return Operating(decided / n, Value(errors / decided, *risk_bounds(errors, decided, ci)), k)


def summed(record: CurveRecord, name: str) -> Counted | None:
    """The calibrated table at the k of the per-check rows ``name``, over every case.

    Those rows split the curve's cases by check and use the same table (``curve_parts``,
    ``compare_tables``), and a case's decision depends on nothing else, so their counts add up
    to the aggregate's; the errors come back exactly from ``risk`` = errors / decided. None when
    a check has no such row, the rows hold different k or other cases than the curve, or a risk
    is no count's ratio."""
    rows = _covering_rows(record, name)
    if rows is None:
        return None
    counted = [e for e in map(_errors, rows) if e is not None]
    if len(counted) != len(rows):
        return None
    n, decided = sum(r.n for r in rows), sum(r.decided for r in rows)
    return Counted(rows[0].k_sigma, n, decided, sum(counted))


def _covering_rows(record: CurveRecord, name: str) -> list[PointRow] | None:
    rows = [checks.get(name) for checks in record.per_check.values()]
    found = [row for row in rows if row is not None]
    one_k = len({row.k_sigma for row in found}) == 1
    cases = {point.n for point in record.tables.calibrated.curve}
    covering = cases == {sum(row.n for row in found)}
    return found if len(found) == len(rows) and one_k and covering else None


def _errors(row: PointRow) -> int | None:
    """The error count behind ``risk`` = errors / decided; None when no count gives it back."""
    if not row.decided:
        return 0
    if not math.isfinite(row.risk):
        return None
    errors = round(row.risk * row.decided)
    return errors if errors / row.decided == row.risk else None


class RiskCoverageFigure:
    name: ClassVar[str] = "risk_coverage"
    inputs: ClassVar[tuple[Output, ...]] = (Output.CURVE,)
    texts: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "x_label": "Coverage (share of clips decided)",
            "y_label": "Selective risk",
            "y_label_estimated": "Selective risk (estimate)",
            CALIBRATED: "Calibrated thresholds",
            RAW: "Raw thresholds",
            "interval": "Risk interval, calibrated",
            CALIBRATION: "Calibration $k^*$ ($k$ = {k})",
            SUMMED: "Calibration $k^*$ ($k$ = {k}), summed over checks",
            CONFIGURED: "Configured $k$ ($k$ = {k})",
            NO_K_STAR: "No calibration $k^*$: no $k$ met the target",
            OFF_GRID: "Calibration $k^*$ ($k$ = {k}): no exact point",
            NOT_EXACT: "Configured $k$: no exact sum over checks",
            UNDECIDED: "Nothing decided at $k$ = {k}",
            "target": "Risk target ({value})",
        }
    )

    def data(self, records: Records) -> RiskCoverage:
        target, ci = records.criteria.risk_target, records.abstention.ci
        return risk_coverage(records.curve(), target, ci)

    def draw(self, data: RiskCoverage, text: Text) -> Canvas:
        canvas, axes = grid(1, 1, HEIGHT)
        ax = axes[0, 0]
        for index, curve in enumerate(data.curves):
            if curve.table == CALIBRATED:
                _band(ax, curve, text("interval"))
            ax.plot(
                curve.coverage,
                [r.value for r in curve.risk],
                label=text(curve.table),
                **series(index).kwargs(),
            )
        _markers(ax, data.markers, text)
        reference(ax, data.target, text("target", value=text.number(data.target)), dash=":")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        numbers(ax.xaxis, text)
        numbers(ax.yaxis, text)
        ax.set_xlabel(text("x_label"))
        ax.set_ylabel(text("y_label_estimated" if data.estimated else "y_label"))
        outside_legend(canvas, [ax], columns=2)
        return canvas


def _band(ax: Axes, curve: Swept, label: str) -> None:
    rows = [
        (coverage, risk.low, risk.high)
        for coverage, risk in zip(curve.coverage, curve.risk, strict=True)
        if risk.low is not None and risk.high is not None
    ]
    if rows:
        coverage, low, high = zip(*rows, strict=True)
        ax.fill_between(coverage, low, high, color=BAND, linewidth=0, label=label)


def _markers(ax: Axes, markers: Sequence[Marker], text: Text) -> None:
    """Each point drawn with its legend entry; a point not drawn gets an entry naming why."""
    for index, marker in enumerate(markers):
        found = marker.found
        if isinstance(found, Gap):
            ax.plot([], [], linestyle="none", label=_named(found.reason, found.k_sigma, text))
            continue
        label = _named(marker.label, found.k_sigma, text)
        with_intervals(ax, [found.coverage], [found.risk], highlight(index), label, line=False)


def _named(key: str, k: float | None, text: Text) -> str:
    return text(key) if k is None else text(key, k=text.number(k, K_DIGITS))


PLUGINS = (RiskCoverageFigure,)
