"""An implemented command: only its help is exercised."""

from __future__ import annotations

import typer

NAME = "ready"
OWNER = "fixture"
IMPLEMENTED = True

app = typer.Typer(help="Implemented fixture command.", no_args_is_help=True)


@app.command("go", help="Print ok.")
def go() -> None:
    typer.echo("ok")
