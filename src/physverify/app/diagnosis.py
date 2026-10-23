"""Wiring of a diagnosis run from the application context."""

from __future__ import annotations

import datetime as dt
import functools

from physverify.app.context import AppContext
from physverify.app.services import dataset_for, pipeline_deps
from physverify.diagnosis.context import DiagnosisContext, sampled_metrics
from physverify.diagnosis.settings import diagnosis_settings
from physverify.diagnosis.verification_adapter import CheckSuite
from physverify.extraction._base import StageObserver
from physverify.runtime.runs import new_run


def diagnosis_context(
    app: AppContext,
    observer: StageObserver,
    suite: CheckSuite | None,
    *,
    dataset: str | None = None,
    subset: str | None = None,
) -> DiagnosisContext:
    settings = app.settings
    overrides = {"dataset": dataset, "subset": subset}
    diag = diagnosis_settings(settings).model_copy(
        update={key: value for key, value in overrides.items() if value is not None}
    )
    clock = functools.partial(dt.datetime.now, dt.UTC)
    run = new_run(settings, app.invocation.argv, clock())
    return DiagnosisContext(
        settings=settings,
        diag=diag,
        dataset_for=functools.partial(dataset_for, app),
        providers=app.registry("providers"),
        checks=app.registry("checks"),
        metrics=sampled_metrics(app.registry("metrics"), diag.metrics),
        deps=pipeline_deps(app, observer),
        suite=suite,
        clock=clock,
        run=run,
    )
