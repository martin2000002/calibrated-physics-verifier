"""The figure protocol, the run outputs a figure may read and the ``figures`` registry."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from physverify.config.plugin_context import PluginKind
from physverify.core.errors import InvalidPluginError, PhysverifyError
from physverify.core.registry import Registry, discover
from physverify.stats.intervals import Rate

if TYPE_CHECKING:
    from matplotlib.figure import Figure as Canvas

    from physverify.reporting.figures._labels import Text
    from physverify.reporting.figures._records import Records

FIGURES_PACKAGE = "physverify.reporting.figures"
CHANCE = 0.5
"""The AUROC and the pair accuracy of a coin flip."""


class Output(StrEnum):
    """A run output; the value is its key in ``run/state.env`` (``CURVE`` is found under runs/)."""

    KUBRIC_EVAL = "KUBRIC_EVAL"
    REAL_EVAL = "REAL_EVAL"
    ATTRIBUTION = "ATTRIBUTION"
    VLM_KUBRIC = "VLM_KUBRIC"
    CURVE = "CURVE"


@dataclass(frozen=True)
class Value:
    """A plotted value and its interval; None where the record has none."""

    value: float | None
    low: float | None = None
    high: float | None = None

    @classmethod
    def of_rate(cls, rate: Rate) -> Value:
        return cls(rate.value, rate.ci.low, rate.ci.high)

    @classmethod
    def finite(cls, value: float | None, low: float | None, high: float | None) -> Value:
        """NaN (a JSON null read back) becomes None."""
        return cls(*(None if v is None or math.isnan(v) else v for v in (value, low, high)))


class NoDataError(PhysverifyError):
    """The records hold nothing the figure can draw; the message says why."""


class MalformedInputError(PhysverifyError):
    """A run output exists but cannot be read or validated; the message names it and says why."""


class Figure[T](Protocol):
    """Registered by ``name`` (also the file name). ``texts`` holds the English default of every
    string it draws. ``data`` reduces the parsed outputs that ``records`` holds (each is read once
    per render) and raises NoDataError when they hold nothing to draw; ``draw`` only lays the data
    out."""

    name: ClassVar[str]
    inputs: ClassVar[tuple[Output, ...]]
    texts: ClassVar[Mapping[str, str]]

    def data(self, records: Records) -> T: ...

    def draw(self, data: T, text: Text) -> Canvas: ...


FIGURES = PluginKind[type[Figure[Any]]]("figures")


def validate_figure(figure: type[Figure[Any]]) -> None:
    if not all(isinstance(key, Output) for key in figure.inputs):
        raise InvalidPluginError(f"{figure.name}: inputs must be Output members")
    if not all(isinstance(v, str) for v in figure.texts.values()):
        raise InvalidPluginError(f"{figure.name}: every text must be a string")


def figures() -> Registry[type[Figure[Any]]]:
    return discover(FIGURES_PACKAGE, (Figure,), "name", validate_figure)
