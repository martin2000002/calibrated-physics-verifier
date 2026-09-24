"""``physverify timing`` (stub owned by cycle A3)."""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.cli.context import not_implemented

NAME = "timing"
OWNER = "A3"
IMPLEMENTED = False

app = typer.Typer(
    help="Timing and throughput measurements. Not implemented yet (cycle A3).", no_args_is_help=True
)


@app.command("run", help="Time a target on this machine or the H200.")
def run(
    target: Annotated[str | None, typer.Option(help="Timing target name.")] = None,
    device: Annotated[str | None, typer.Option(help="cpu or cuda.")] = None,
) -> None:
    not_implemented(OWNER, "timing run")
