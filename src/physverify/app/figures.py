"""The ``report figures`` service: every registered figure whose inputs a finished run wrote."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.services import excluded_checks
from physverify.baselines.settings import load_vlm_comparison
from physverify.config.plugin_context import registry_of
from physverify.core.output import Echo
from physverify.reporting.figures._base import FIGURES
from physverify.reporting.figures._labels import load_labels
from physverify.reporting.figures._publish import (
    Outcome,
    label_defaults,
    render_figures,
    select,
    succeeded,
)
from physverify.reporting.figures._records import Records, locate
from physverify.selective_risk.settings import SECTION, AbstentionSettings

__all__ = ["FigureRequest", "Outcome", "succeeded", "write_figures"]


@dataclass(frozen=True)
class FigureRequest:
    final_dir: Path
    """The final run's directory: ``run/state.env`` and ``runs/*/abstention_report.json``."""
    out: Path
    labels: Path | None = None
    only: tuple[str, ...] = ()


def write_figures(context: AppContext, request: FigureRequest, echo: Echo) -> tuple[Outcome, ...]:
    registry = registry_of(context, FIGURES)
    labels = load_labels(request.labels, label_defaults(registry))
    records = Records(
        locate(request.final_dir),
        load_vlm_comparison(context.settings),
        excluded_checks(context),
        abstention=context.settings.section(SECTION, AbstentionSettings),
    )
    outcomes = render_figures(select(registry, request.only), records, labels, request.out)
    for outcome in outcomes:
        echo(outcome.line())
    return outcomes
