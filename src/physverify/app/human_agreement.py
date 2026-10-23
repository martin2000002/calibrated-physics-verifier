"""Human-agreement wiring: the VideoPhy label source, the subset review and the comparison
behind ``physverify agreement``."""

from __future__ import annotations

import datetime as dt
import functools
from pathlib import Path

from physverify.app.context import AppContext
from physverify.core.output import Echo
from physverify.datasets._base import fetch_all
from physverify.datasets.videophy import VideophyDataset
from physverify.human_agreement.clip_decision import import_calibration
from physverify.human_agreement.settings import agreement_settings
from physverify.human_agreement.subset_review import check_recorded, record_subsets, review_subset
from physverify.human_agreement.workflow import CompareRequest as CompareRequest
from physverify.human_agreement.workflow import apply_freeze, compare_and_report, freeze


def _source(context: AppContext) -> VideophyDataset:
    return VideophyDataset(context.settings)


def fetch(context: AppContext, subset: str | None, *, echo: Echo) -> None:
    fetch_all([VideophyDataset], context.settings, subset, echo=echo)


def subset(
    context: AppContext, name: str | None, *, show: bool, record: bool, check: bool, echo: Echo
) -> None:
    """Check the recorded review, record it, or write (and with ``show`` print) the pending one."""
    dataset = _source(context)
    chosen = name or agreement_settings(context.settings).test_subset
    if check:
        check_recorded(dataset, echo=echo)
    elif record:
        record_subsets(dataset, chosen, echo=echo)
    else:
        review_subset(dataset, context.settings, chosen, show=show, echo=echo)


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
        context.registry("checks"),
        _source(context),
        run_dir,
        import_calibration,
        dt.datetime.now(dt.UTC),
    )
    apply_freeze(path, current, force=force, deviation=deviation, echo=echo, err=err)


def compare(context: AppContext, request: CompareRequest, *, echo: Echo) -> None:
    compare_and_report(
        context.settings,
        context.registry("checks"),
        _source(context),
        request,
        import_calibration,
        functools.partial(dt.datetime.now, dt.UTC),
        echo=echo,
    )
