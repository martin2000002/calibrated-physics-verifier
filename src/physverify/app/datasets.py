"""The ``download`` service: which real datasets to fetch, and the fetch itself."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from physverify.app.context import AppContext
from physverify.config.plugin_context import registry_of
from physverify.core.errors import UsageError
from physverify.core.output import Echo
from physverify.datasets._base import DATASETS, fetch_all
from physverify.diagnosis.settings import diagnosis_settings


def fetch_targets(
    registry: Mapping[str, type[Any]],
    downloadable: Sequence[str],
    dataset: str | None,
    by_name: Sequence[str] = (),
) -> list[type[Any]]:
    """The dataset classes to fetch: ``dataset``, or every downloadable one when it is None.
    ``by_name`` datasets are fetched only when named."""
    allowed = [*downloadable, *by_name]
    if dataset is not None and dataset not in allowed:
        raise UsageError(f"{dataset!r} cannot be downloaded; downloadable: {allowed}")
    names = downloadable if dataset is None else (dataset,)
    unregistered = [name for name in names if name not in registry]
    if unregistered:
        raise UsageError(f"datasets {unregistered} are not registered")
    return [registry[name] for name in names]


def fetch_real(context: AppContext, dataset: str | None, subset: str | None, *, echo: Echo) -> None:
    """Fetch ``dataset``, or every one of ``diagnosis.real.datasets``; the sealed controls of
    ``diagnosis.real.sealed_controls`` only when named."""
    settings = context.settings
    real = diagnosis_settings(settings).real
    registry = registry_of(context, DATASETS)
    classes = fetch_targets(registry, real.datasets, dataset, real.sealed_controls)
    fetch_all(classes, settings, subset, echo=echo)
