"""``physverify report``: publication output of a finished run.

Read only: ``figures`` reads the outputs the final run wrote and selects no dataset split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physverify.app.figures import FigureRequest, succeeded, write_figures
from physverify.cli.context import app_context

NAME = "report"

app = typer.Typer(help="Publication output of a finished run.")


@app.command("figures", help="Draw the thesis figures (one PDF each) from a finished run.")
def figures(
    ctx: typer.Context,
    final_dir: Annotated[
        Path, typer.Option(help="The final run's directory (holds run/state.env).")
    ],
    out: Annotated[Path, typer.Option(help="Output directory for the PDFs.")],
    labels: Annotated[
        Path | None, typer.Option(help="TOML of figure texts replacing the English defaults.")
    ] = None,
    only: Annotated[
        list[str] | None, typer.Option(help="Draw only this figure (repeatable).")
    ] = None,
) -> None:
    request = FigureRequest(final_dir, out, labels, tuple(only or ()))
    outcomes = write_figures(app_context(ctx), request, typer.echo)
    if not succeeded(outcomes):
        raise typer.Exit(code=1)
