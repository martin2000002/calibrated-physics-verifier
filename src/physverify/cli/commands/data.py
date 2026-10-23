"""``physverify data fetch``: download the real datasets into ``paths.raw_dir`` (ADR 0007 §4.3).

A heavy command (network and disk). A second run with the files intact transfers nothing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from physverify.cli.context import app_context, heavy_command
from physverify.core.download import BYTES_PER_MB, MANIFEST_FILE, read_manifest
from physverify.core.errors import UsageError
from physverify.diagnosis.settings import diagnosis_settings

if TYPE_CHECKING:
    from physverify.config.models import Settings

NAME = "data"

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


def fetch_summary(dest: Path, elapsed_s: float) -> str:
    manifest = read_manifest(dest / MANIFEST_FILE)
    if manifest is None:
        return f"{dest.name}: no manifest written ({elapsed_s:.1f} s)"
    size_mb = sum(f.size for f in manifest.files) / BYTES_PER_MB
    return (
        f"{manifest.dataset} [{manifest.subset}]: {len(manifest.clips)} clips, "
        f"{len(manifest.files)} files, {size_mb:.1f} MB, {elapsed_s:.1f} s -> {dest}"
    )


def fetch_all(
    classes: Sequence[type[Any]],
    settings: Settings,
    subset: str | None,
    echo: Callable[[str], None],
    clock: Callable[[], float] = time.monotonic,
) -> None:
    for cls in classes:
        dest = settings.paths.raw_dir / cls.name
        started = clock()
        cls(settings).fetch(subset, dest)
        echo(fetch_summary(dest, clock() - started))


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
    fetch_all(classes, settings, subset, typer.echo)
