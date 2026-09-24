"""``physverify data`` (stub owned by cycle A6)."""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.cli.context import not_implemented

NAME = "data"
OWNER = "A6"
IMPLEMENTED = False

app = typer.Typer(
    help="Download real datasets. Not implemented yet (cycle A6).", no_args_is_help=True
)


@app.command("fetch", help="Fetch physics_iq or pisa_real.")
def fetch(
    dataset: Annotated[str | None, typer.Argument(help="physics_iq or pisa_real.")] = None,
) -> None:
    not_implemented(OWNER, "data fetch")
