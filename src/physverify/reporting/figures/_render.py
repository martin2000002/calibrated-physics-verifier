"""The shared look of the figures: text width, type sizes, grayscale series and the PDF file.

matplotlib is imported inside the functions: building the registry never loads it."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from physverify.core.errors import UsageError
from physverify.reporting.figures._base import Value
from physverify.schema.checks import InputLevel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.container import ErrorbarContainer
    from matplotlib.figure import Figure as Canvas
    from matplotlib.typing import LegendLocType, RcKeyType

    from physverify.reporting.figures._labels import Text

TEXT_WIDTH = 6.0
"""Inches: the width of the thesis text block."""
PDF = "pdf"
STYLE: Mapping[RcKeyType, Any] = MappingProxyType(
    {
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "figure.labelsize": 8.0,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "errorbar.capsize": 2.0,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)
"""pdf.fonttype 42 embeds TrueType fonts, so the text stays text."""


@dataclass(frozen=True)
class Series:
    """One series, told apart in grayscale by marker and line, not by hue."""

    color: str
    marker: str
    linestyle: str
    filled: bool = True
    size: float | None = None

    def kwargs(self, *, line: bool = True) -> dict[str, Any]:
        found: dict[str, Any] = {
            "color": self.color,
            "marker": self.marker,
            "linestyle": self.linestyle if line else "none",
            "markerfacecolor": self.color if self.filled else "white",
            "markeredgecolor": self.color,
        }
        return found if self.size is None else found | {"markersize": self.size}


SERIES = (
    Series("0.0", "o", "-"),
    Series("0.45", "s", "--", filled=False),
    Series("0.25", "^", ":"),
    Series("0.6", "D", "-.", filled=False),
)
HIGHLIGHTS = (
    Series("0.0", "*", "-", size=10.0),
    Series("0.0", "D", "-", filled=False, size=6.0),
)
"""Chosen points, such as the operating points of a curve."""
STRATA = tuple(level.value for level in InputLevel)
REFERENCE: Mapping[str, Any] = MappingProxyType({"color": "0.4", "linewidth": 0.8})
"""A reference line: chance, or the limit of a criterion."""
ROUND_TICK_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)
"""Tick steps per decade of a capped axis: the ones matplotlib's default locator uses."""
BAR: Mapping[str, Any] = MappingProxyType({"color": "0.8", "edgecolor": "0.0", "linewidth": 0.6})
MISSING: Mapping[str, Any] = MappingProxyType(
    {"color": "0.3", "fontsize": 7.0, "fontstyle": "italic", "ha": "left", "va": "center"}
)
"""A missing value, named in its row instead of drawn as a zero."""


def series(index: int) -> Series:
    return SERIES[index % len(SERIES)]


def highlight(index: int) -> Series:
    return HIGHLIGHTS[index % len(HIGHLIGHTS)]


def stratum_series(level: str) -> Series:
    """The same look for an input level in every figure."""
    return series(STRATA.index(level) if level in STRATA else len(STRATA))


def require_matplotlib() -> None:
    if importlib.util.find_spec("matplotlib") is None:
        raise UsageError("matplotlib is not installed: install the extra (uv sync --extra figures)")


def grid(rows: int, cols: int, row_height: float, **subplots: Any) -> tuple[Canvas, Any]:
    """A text-wide figure and its (rows, cols) array of axes."""
    from matplotlib.figure import Figure as Canvas

    canvas = Canvas(figsize=(TEXT_WIDTH, rows * row_height), layout="constrained")
    return canvas, canvas.subplots(rows, cols, squeeze=False, **subplots)


def save(draw: Callable[[], Canvas], path: Path) -> None:
    """Draw and write one PDF under the shared style, without a creation date: the same data
    give the same file."""
    import matplotlib

    with matplotlib.rc_context(dict(STYLE)):
        draw().savefig(path, format=PDF, metadata={"CreationDate": None})


def numbers(axis: Axis, text: Text, *, most: int | None = None) -> None:
    """Tick labels of a numeric axis with the labels' decimal sign; ``most`` caps the number of
    tick intervals, for panels too narrow for the default spacing, on the default locator's round
    steps."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    if most is not None:
        axis.set_major_locator(MaxNLocator(nbins=most, steps=ROUND_TICK_STEPS))
    axis.set_major_formatter(FuncFormatter(lambda value, _: text.tick(value)))


def with_intervals(
    axes: Axes,
    positions: Sequence[float],
    values: Sequence[Value],
    style: Series,
    label: str,
    *,
    horizontal: bool = False,
    line: bool = True,
) -> ErrorbarContainer:
    """Points with their intervals; a missing value or bound draws nothing for it."""
    middle = [_nan(v.value) for v in values]
    below = [_spread(m, v.low, m) for m, v in zip(middle, values, strict=True)]
    above = [_spread(m, m, v.high) for m, v in zip(middle, values, strict=True)]
    spread = [below, above]
    if horizontal:
        return axes.errorbar(middle, positions, xerr=spread, label=label, **style.kwargs(line=line))
    return axes.errorbar(positions, middle, yerr=spread, label=label, **style.kwargs(line=line))


def bars(axes: Axes, places: Sequence[float], values: Sequence[float | None], missing: str) -> None:
    """Horizontal bars; a missing value is named by ``missing`` at the start of its row."""
    known = [
        (place, value) for place, value in zip(places, values, strict=True) if value is not None
    ]
    axes.barh([place for place, _ in known], [value for _, value in known], **BAR)
    for place, value in zip(places, values, strict=True):
        if value is None:
            axes.annotate(missing, (0.02, place), xycoords=("axes fraction", "data"), **MISSING)


def category_rows(axes: Axes, labels: Sequence[str]) -> None:
    """Rows top to bottom, each inside the axes even when nothing is drawn in it."""
    axes.set_yticks(range(len(labels)), labels)
    axes.set_ylim(len(labels) - 0.5, -0.5)


def reference(
    axes: Axes, at: float, label: str, *, vertical: bool = False, dash: str = "--"
) -> None:
    draw = axes.axvline if vertical else axes.axhline
    draw(at, linestyle=dash, label=label, **REFERENCE)


def outside_legend(
    canvas: Canvas,
    axes: Sequence[Axes],
    columns: int = 4,
    loc: LegendLocType = "outside lower center",
) -> None:
    """One legend outside the axes (under them by default) with every distinct label of ``axes``."""
    found: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        found.update({lab: h for h, lab in zip(handles, labels, strict=True) if lab not in found})
    if found:
        canvas.legend(list(found.values()), list(found), loc=loc, ncols=columns)


def _nan(value: float | None) -> float:
    return math.nan if value is None else value


def _spread(middle: float, low: float | None, high: float | None) -> float:
    if low is None or high is None or math.isnan(middle):
        return 0.0
    return max(high - low, 0.0)
