"""``physverify verify`` (stub owned by cycle A7)."""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.cli.context import not_implemented

NAME = "verify"
OWNER = "A7"
IMPLEMENTED = False

app = typer.Typer(
    invoke_without_command=True,
    help="Run the physics checks R1-R5 over a dataset. Not implemented yet (cycle A7).",
)


@app.callback()
def verify(
    dataset: Annotated[str | None, typer.Option(help="Dataset name.")] = None,
    level: Annotated[str | None, typer.Option(help="oracle or perceived.")] = None,
    gt: Annotated[list[str] | None, typer.Option(help="Stages replaced by ground truth.")] = None,
) -> None:
    not_implemented(OWNER, "verify")
