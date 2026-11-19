"""The verifier against the VLM judge on the Kubric part: pair accuracy (a perturbed clip scored
above its scene's clean clip) of B1 and of the judge, and their difference, each with its
whole-scene bootstrap interval."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from physverify.baselines.records import PartRows
from physverify.baselines.settings import BootstrapSettings
from physverify.baselines.stats import Estimate
from physverify.reporting.figures._base import CHANCE, NoDataError, Output, Value
from physverify.reporting.figures._render import (
    grid,
    numbers,
    outside_legend,
    reference,
    series,
    with_intervals,
)
from physverify.reporting.final_comparison import pair_accuracies

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure as Canvas

    from physverify.reporting.figures._labels import Text
    from physverify.reporting.figures._records import Records

KUBRIC = "kubric"
HEIGHT = 1.5
WIDTHS = (2, 1)
POINT = series(0)


@dataclass(frozen=True)
class PairAccuracy:
    judge: str
    b1: Value
    judged: Value
    delta: Value
    n_pairs: int
    unread: float
    """Share of the part's clips the judge left without a readable score."""


def pair_accuracy(part: PartRows, boot: BootstrapSettings) -> PairAccuracy:
    if part.part != KUBRIC:
        raise NoDataError(
            f"pair accuracy needs the Kubric part; the rows are the {part.part!r} part"
        )
    found = pair_accuracies(part, boot)
    n_pairs = found.estimates["delta"].n
    if not n_pairs:
        raise NoDataError("no scene has one clean clip and a perturbed one")
    values = {key: _value(found.estimates[key]) for key in ("b1", "judge", "delta")}
    return PairAccuracy(
        part.judge, values["b1"], values["judge"], values["delta"], n_pairs, found.unread
    )


def _value(estimate: Estimate) -> Value:
    return Value(estimate.value, estimate.ci.low, estimate.ci.high)


class PairAccuracyFigure:
    name: ClassVar[str] = "pair_accuracy"
    inputs: ClassVar[tuple[Output, ...]] = (Output.VLM_KUBRIC,)
    texts: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "x_label": "Pair accuracy over {n} clean-perturbed pairs",
            "delta_label": "Difference",
            "b1": "B1 (verifier)",
            "judge": "{judge} ({unread}% unread)",
            "delta": "B1 minus judge",
            "chance": "Chance ({value})",
            "zero": "No difference",
        }
    )

    def data(self, records: Records) -> PairAccuracy:
        return pair_accuracy(records.kubric_rows(), records.comparison.pairs)

    def draw(self, data: PairAccuracy, text: Text) -> Canvas:
        canvas, axes = grid(1, 2, HEIGHT, gridspec_kw={"width_ratios": WIDTHS})
        _accuracy(axes[0, 0], data, text)
        _difference(axes[0, 1], data, text)
        outside_legend(canvas, list(axes[0, :]), columns=2)
        return canvas


def _accuracy(ax: Axes, data: PairAccuracy, text: Text) -> None:
    judge = text("judge", judge=text.name("judges", data.judge), unread=_percent(data, text))
    with_intervals(ax, [0, 1], [data.b1, data.judged], POINT, "", horizontal=True, line=False)
    reference(ax, CHANCE, text("chance", value=text.number(CHANCE, 1)), vertical=True, dash=":")
    ax.set_yticks([0, 1], [text("b1"), judge])
    ax.set_ylim(1.6, -0.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(text("x_label", n=data.n_pairs))
    numbers(ax.xaxis, text)


def _difference(ax: Axes, data: PairAccuracy, text: Text) -> None:
    with_intervals(ax, [0], [data.delta], POINT, "", horizontal=True, line=False)
    reference(ax, 0.0, text("zero"), vertical=True)
    ax.set_yticks([0], [text("delta")])
    ax.set_ylim(0.6, -0.6)
    ax.set_xlabel(text("delta_label"))
    numbers(ax.xaxis, text)


def _percent(data: PairAccuracy, text: Text) -> str:
    return text.number(100 * data.unread, 0)


PLUGINS = (PairAccuracyFigure,)
