"""``physverify doctor``: show the resolved environment and optionally probe the heavy lock."""

from __future__ import annotations

import time
from typing import Annotated

import typer

from physverify.app import doctor as service
from physverify.cli.context import app_context

NAME = "doctor"

app = typer.Typer(
    invoke_without_command=True,
    help="Show configuration, paths, lock and package versions.",
)


@app.callback()
def doctor(
    ctx: typer.Context,
    lock_probe: Annotated[
        float | None,
        typer.Option(help="Hold the heavy lock for this many seconds (tests the lock)."),
    ] = None,
    probe_container: Annotated[
        bool,
        typer.Option(help="Import the Kubric build's modules in the container, with its mounts."),
    ] = False,
) -> None:
    context = app_context(ctx)
    for line in service.report_lines(context):
        typer.echo(line)
    if probe_container:
        typer.echo(service.container_probe(context))
    if lock_probe is None:
        return
    with context.heavy_lock():
        typer.echo(f"lock-probe: holding the heavy lock for {lock_probe:g} s")
        time.sleep(lock_probe)
    typer.echo("lock-probe: released")
