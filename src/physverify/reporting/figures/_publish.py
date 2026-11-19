"""Render the selected figures of one run: each whose inputs exist and parse. A figure whose
inputs are missing or empty is skipped, one whose input is malformed fails; both with the reason,
and the others are still drawn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from physverify.core.errors import UsageError
from physverify.reporting.figures._base import Figure, MalformedInputError, NoDataError
from physverify.reporting.figures._labels import COMMON, COMMON_TEXTS, Labels
from physverify.reporting.figures._records import Records
from physverify.reporting.figures._render import PDF, require_matplotlib, save

type Figures = Mapping[str, type[Figure[Any]]]


@dataclass(frozen=True)
class Outcome:
    name: str
    path: Path | None = None
    skipped: str = ""
    failed: str = ""

    def line(self) -> str:
        if self.path is not None:
            return f"figure {self.name}: {self.path}"
        if self.failed:
            return f"failed {self.name}: {self.failed}"
        return f"skipped {self.name}: {self.skipped}"


def succeeded(outcomes: Sequence[Outcome]) -> bool:
    """At least one figure drawn and none failed."""
    return any(o.path for o in outcomes) and not any(o.failed for o in outcomes)


def select(registry: Figures, only: Sequence[str]) -> Figures:
    """The figures named in ``only`` (all of them when empty), in registry order."""
    unknown = sorted(set(only) - set(registry))
    if unknown:
        raise UsageError(f"unknown figures {unknown}; known: {list(registry)}")
    return {name: figure for name, figure in registry.items() if not only or name in only}


def label_defaults(registry: Figures) -> dict[str, Mapping[str, str]]:
    return {COMMON: COMMON_TEXTS, **{name: figure.texts for name, figure in registry.items()}}


def render_figures(
    figures: Figures, records: Records, labels: Labels, out: Path
) -> tuple[Outcome, ...]:
    require_matplotlib()
    out.mkdir(parents=True, exist_ok=True)
    return tuple(_render(figure(), records, labels, out) for figure in figures.values())


def _render(figure: Figure[Any], records: Records, labels: Labels, out: Path) -> Outcome:
    gaps = [gap for key in figure.inputs if (gap := records.gap(key)) is not None]
    if gaps:
        return Outcome(figure.name, skipped="; ".join(gaps))
    try:
        data = figure.data(records)
    except NoDataError as error:
        return Outcome(figure.name, skipped=str(error))
    except MalformedInputError as error:
        return Outcome(figure.name, failed=str(error))
    path = out / f"{figure.name}.{PDF}"
    text = labels.scope(figure.name)
    save(lambda: figure.draw(data, text), path)
    return Outcome(figure.name, path)
