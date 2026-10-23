"""``physverify download fetch``: download the real datasets into ``paths.raw_dir``.

A heavy command (network and disk). A second run with the files intact transfers nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

import typer

from physverify.cli.context import app_context, heavy_command
from physverify.core.errors import UsageError
from physverify.datasets._base import fetch_all
from physverify.diagnosis.settings import diagnosis_settings

NAME = "download"

app = typer.Typer(help="Download the real datasets (Physics-IQ, PISA).", no_args_is_help=True)


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


@app.command(
    "fetch",
    help="Fetch the real datasets (default: diagnosis.real.datasets; the sealed controls "
    "diagnosis.real.sealed_controls only by name).",
)
@heavy_command
def fetch(
    dataset: Annotated[
        str | None,
        typer.Argument(help="A real dataset or sealed control (default: diagnosis.real.datasets)."),
    ] = None,
    subset: Annotated[str | None, typer.Option(help="local or full (default: local).")] = None,
) -> None:
    context = app_context()
    settings = context.settings
    real = diagnosis_settings(settings).real
    registry = context.registry("datasets")
    classes = fetch_targets(registry, real.datasets, dataset, real.sealed_controls)
    fetch_all(classes, settings, subset, echo=typer.echo)
