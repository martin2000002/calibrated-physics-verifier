"""Human-agreement wiring: the configured label source, its subset review and the comparison
behind ``physverify agreement``."""

from __future__ import annotations

import datetime as dt
import functools
from pathlib import Path

from physverify.app.context import AppContext
from physverify.app.services import label_source_class
from physverify.calibration.api import calibration_api
from physverify.checks._base import CHECKS
from physverify.config.plugin_context import PluginContext, registry_of
from physverify.core.output import Echo
from physverify.datasets._base import HumanLabelSource, fetch_all
from physverify.human_agreement.settings import SECTION, agreement_settings
from physverify.human_agreement.workflow import CompareRequest as CompareRequest
from physverify.human_agreement.workflow import apply_freeze, compare_and_report, freeze

LABEL_SOURCE_KEY = f"[{SECTION}] label_source"


def label_source_type(context: PluginContext) -> type[HumanLabelSource]:
    """The dataset plugin ``[agreement] label_source`` names."""
    name = agreement_settings(context.settings).label_source
    return label_source_class(context, LABEL_SOURCE_KEY, name)


def label_source(context: PluginContext) -> HumanLabelSource:
    return label_source_type(context)(context.settings)


def fetch(context: AppContext, subset: str | None, *, echo: Echo) -> None:
    fetch_all([label_source_type(context)], context.settings, subset, echo=echo)


def subset(
    context: AppContext, name: str | None, *, show: bool, record: bool, check: bool, echo: Echo
) -> None:
    """Check the recorded review, record it, or write (and with ``show`` print) the pending one."""
    dataset = label_source(context)
    chosen = name or agreement_settings(context.settings).test_subset
    if check:
        dataset.check_review(echo=echo)
    elif record:
        dataset.record_review(chosen, echo=echo)
    else:
        dataset.review_subset(chosen, show=show, echo=echo)


def preregister(
    context: AppContext,
    run_dir: Path,
    *,
    force: bool,
    deviation: str | None,
    echo: Echo,
    err: Echo,
) -> None:
    path, current = freeze(
        context.settings,
        registry_of(context, CHECKS),
        label_source(context),
        run_dir,
        calibration_api,
        dt.datetime.now(dt.UTC),
    )
    apply_freeze(path, current, force=force, deviation=deviation, echo=echo, err=err)


def compare(context: AppContext, request: CompareRequest, *, echo: Echo) -> None:
    compare_and_report(
        context.settings,
        registry_of(context, CHECKS),
        label_source(context),
        request,
        calibration_api,
        functools.partial(dt.datetime.now, dt.UTC),
        echo=echo,
    )
