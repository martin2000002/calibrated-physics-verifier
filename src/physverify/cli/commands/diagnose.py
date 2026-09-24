"""``physverify diagnose`` (stub owned by cycle A6)."""

from __future__ import annotations

from typing import Annotated

import typer

from physverify.cli.context import not_implemented

NAME = "diagnose"
OWNER = "A6"
IMPLEMENTED = False

app = typer.Typer(
    help="Stage error profiles, end-to-end error and attribution. Not implemented yet (cycle A6).",
    no_args_is_help=True,
)


@app.command("stages", help="Per-stage error against ground truth.")
def stages(
    write_profile: Annotated[bool, typer.Option(help="Write the error profile.")] = False,
) -> None:
    not_implemented(OWNER, "diagnose stages")


@app.command("end-to-end", help="End-to-end verdict error.")
def end_to_end() -> None:
    not_implemented(OWNER, "diagnose end-to-end")


@app.command("attribution", help="Factorial attribution of errors to stages.")
def attribution() -> None:
    not_implemented(OWNER, "diagnose attribution")


@app.command("real", help="Diagnosis on real annotated clips.")
def real() -> None:
    not_implemented(OWNER, "diagnose real")
