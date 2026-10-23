"""``physverify download fetch``: download the real datasets into ``paths.raw_dir``.

A heavy command (network and disk). A second run with the files intact transfers nothing.
"""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.app.datasets import fetch_real
from physverify.cli.context import app_context, heavy_command

NAME = "download"

app = typer.Typer(help="Download the real datasets (Physics-IQ, PISA).", no_args_is_help=True)


@app.command(
    "fetch",
    help="Fetch the real datasets (default: diagnosis.real.datasets; the sealed controls "
    "diagnosis.real.sealed_controls only by name).",
)
@heavy_command
def fetch(
    ctx: typer.Context,
    dataset: Annotated[
        str | None,
        typer.Argument(help="A real dataset or sealed control (default: diagnosis.real.datasets)."),
    ] = None,
    subset: Annotated[str | None, typer.Option(help="local or full (default: local).")] = None,
) -> None:
    fetch_real(app_context(ctx), dataset, subset, echo=typer.echo)
